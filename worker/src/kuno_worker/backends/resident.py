"""Keeping model weights loaded between jobs.

Inside a confidential VM, loading weights is up to 34x slower than normal (encrypted
bounce buffers), and LTX-2.5 is ~66 GB while H3 is ~124 GB. A process per job spends
minutes loading and seconds generating, so the pipelines stay resident and jobs queue
behind a lock instead.

The store knows nothing about torch: it takes a loader callable, which lets every rule
here be tested without a GPU.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from kuno_protocol.profiles import ModelProfile

log = logging.getLogger("kuno.worker.resident")


class ModelStore:
    """Loads a profile's pipeline once and keeps it; evicts the least recently used one
    when a worker serves more profiles than fit in VRAM."""

    def __init__(self, loader: Callable[[ModelProfile], Any], capacity: int = 1):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._loader = loader
        self._capacity = capacity
        self._pipelines: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()
        self.loads = 0
        self.evictions = 0

    @property
    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._pipelines)

    def warm(self, profile: ModelProfile) -> None:
        """Load ahead of the first job, so no customer waits for weights."""
        with self._lock:
            self._get(profile)

    def _get(self, profile: ModelProfile) -> Any:
        pipeline = self._pipelines.get(profile.id)
        if pipeline is not None:
            self._pipelines.move_to_end(profile.id)
            return pipeline
        while len(self._pipelines) >= self._capacity:
            evicted_id, evicted = self._pipelines.popitem(last=False)
            self.evictions += 1
            log.info("evicting %s to make room for %s", evicted_id, profile.id)
            _release(evicted)
        started = time.time()
        log.info("loading %s (%s, %g GB per GPU)", profile.id, profile.checkpoint, profile.min_vram_gb)
        pipeline = self._loader(profile)
        self.loads += 1
        self._pipelines[profile.id] = pipeline
        log.info("loaded %s in %.1fs", profile.id, time.time() - started)
        return pipeline

    @contextmanager
    def acquire(self, profile: ModelProfile) -> Iterator[Any]:
        """One generation at a time: the GPUs are the bottleneck, not the queue."""
        with self._lock:
            yield self._get(profile)

    @contextmanager
    def acquire_loaded(self, profile: ModelProfile) -> Iterator[Any]:
        """The most recently used loaded pipeline, whichever profile it serves, or `profile`'s when none is loaded; under
        the same lock as `acquire`. For work every loaded pipeline can do alike, such as writing a plan with the prompt
        enhancer every LTX-2.5 recipe includes, so it never forces a reload."""
        with self._lock:
            if self._pipelines:
                yield next(reversed(self._pipelines.values()))
            else:
                yield self._get(profile)

    def unload_all(self) -> None:
        with self._lock:
            while self._pipelines:
                _, pipeline = self._pipelines.popitem()
                _release(pipeline)


def _release(pipeline: Any) -> None:
    close = getattr(pipeline, "unload", None) or getattr(pipeline, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # a failed unload must not take the worker down
            log.warning("pipeline unload failed", exc_info=False)
    del pipeline
    gc.collect()


class PipelineResult:
    """What a resident pipeline hands back, normalized.

    diffusers returns frames (and, for these models, audio) rather than an encoded file.
    H3's modular pipeline returns a mapping with `videos`, `audio` and `sampling_rate`;
    LTX returns an object with `.frames`. Both shapes are accepted.
    """

    def __init__(self, frames: Any, audio: Any = None, sample_rate: int = 48000):
        self.frames = frames
        self.audio = audio
        self.sample_rate = sample_rate

    @classmethod
    def from_pipeline(cls, raw: Any) -> PipelineResult:
        if isinstance(raw, PipelineResult):
            return raw
        if isinstance(raw, dict):
            frames = raw.get("videos", raw.get("frames"))
            audio, rate = raw.get("audio"), int(raw.get("sampling_rate", 48000))
        else:
            frames = getattr(raw, "frames", None)
            audio = getattr(raw, "audio", None)
            rate = int(getattr(raw, "sampling_rate", 48000))
        if frames is None:
            raise TypeError(f"pipeline returned no frames (got {type(raw).__name__})")
        # Pipelines batch: a single request comes back as a list of one clip.
        if len(frames) and not _looks_like_frame(frames[0]):
            frames = frames[0]
        if audio is not None and len(audio) and not _looks_like_samples(audio):
            audio = audio[0]
        return cls(frames, audio, rate)


def _looks_like_frame(item: Any) -> bool:
    return hasattr(item, "size") or (hasattr(item, "shape") and len(getattr(item, "shape")) == 3)


def _looks_like_samples(item: Any) -> bool:
    shape = getattr(item, "shape", None)
    return shape is not None and len(shape) <= 2
