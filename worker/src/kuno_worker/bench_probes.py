"""What kuno-bench measures with, real or simulated: the machine, a clock, peak memory and per-step timing.

`--backend mock` swaps the model loader for `SimulatedPipeline`, which runs on CPU inside the same resident
backends and advances the bench clock by a plausible GPU time instead of computing. The results have the
real JSON shape with simulated numbers, and say `"simulated": true`.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from kuno_protocol.profiles import FAMILY_H3, FAMILY_LTX, ModelProfile, ltx_num_frames

log = logging.getLogger("kuno.bench")

GIB = 2**30
PACKAGES = ("kuno-worker", "kuno-protocol", "torch", "torchaudio", "torchao", "diffusers", "transformers", "accelerate", "av", "sglang")


class Clock:
    """Seconds for every measurement. Real runs never advance it; simulated pipelines add the GPU time they stand for."""

    def __init__(self) -> None:
        self.offset = 0.0

    def now(self) -> float:
        return time.perf_counter() + self.offset

    def advance(self, seconds: float) -> None:
        self.offset += seconds


class StepTimer:
    """Durations of the denoiser calls during one generation."""

    def __init__(self) -> None:
        self.durations: list[float] = []

    def reset(self) -> None:
        self.durations = []

    def record(self, seconds: float) -> None:
        self.durations.append(seconds)


# ------------------------------------------------------------------ the machine


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    if versions["torch"]:
        try:
            import torch

            versions["cuda"] = torch.version.cuda
        except Exception:  # a broken torch install is reported by the load, not here
            versions["cuda"] = None
    return versions


def probe_machine() -> tuple[Any, dict[str, Any]]:
    """(preflight Host, the `machine` block): GPUs from nvidia-smi and their confidential-computing mode from
    `nvidia-smi conf-compute -f` where the driver has it (None otherwise)."""
    from .backends.quantized import host_memory_gib
    from .preflight import _run, probe

    host = probe()
    cc_query = _run(["nvidia-smi", "conf-compute", "-f"]) or None
    machine = {
        "simulated": False,
        "gpu_model": host.gpus[0].name if host.gpus else None,
        "gpu_count": len(host.gpus),
        "gpus": [{"index": g.index, "name": g.name, "memory_gib": g.memory_gb} for g in host.gpus],
        "driver": host.nvidia_driver,
        "cc_mode": host.gpus[0].cc_mode if host.gpus else None,
        "cc_query": cc_query,
        "cpu_model": host.cpu_model,
        "kernel": host.kernel,
        "host_ram_gib": _round(host_memory_gib()),
        "python": host.python,
        "packages": package_versions(),
    }
    return host, machine


def simulated_machine(gpu_name: str, gpu_count: int, memory_gib: float = 139.8) -> dict[str, Any]:
    import platform

    return {
        "simulated": True,
        "gpu_model": gpu_name,
        "gpu_count": gpu_count,
        "gpus": [{"index": i, "name": gpu_name, "memory_gib": memory_gib} for i in range(gpu_count)],
        "driver": None,
        "cc_mode": None,
        "cc_query": None,
        "cpu_model": "simulated",
        "kernel": platform.release(),
        "host_ram_gib": SIMULATED_HOST_RAM_GIB,
        "python": platform.python_version(),
        "packages": package_versions(),
    }


# ------------------------------------------------------------------ memory


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def _nvml_handles() -> list[Any] | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception:  # no nvidia-ml-py, no driver or no GPU: fall back to nvidia-smi
        return None


def _gpu_used_gib(handles: list[Any] | None) -> list[float]:
    if handles is not None:
        import pynvml

        return [pynvml.nvmlDeviceGetMemoryInfo(handle).used / GIB for handle in handles]
    from .preflight import _run

    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return [float(value) / 1024 for value in out.split()] if out else []


def _host_used_gib() -> float | None:
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        total, available = (int(fields[key].split()[0]) for key in ("MemTotal", "MemAvailable"))
        return (total - available) * 1024 / GIB
    except (OSError, KeyError, ValueError):
        return None


def _reset_peak_rss() -> None:
    try:  # "5" resets VmHWM, the process's peak resident set (Linux)
        Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        pass


def _peak_rss_gib() -> float | None:
    try:
        match = re.search(r"^VmHWM:\s+(\d+) kB", Path("/proc/self/status").read_text(), re.M)
    except OSError:
        return None
    return int(match.group(1)) * 1024 / GIB if match else None


class MemoryMonitor:
    """Peaks while a measurement runs:
    peak_gpu_gib              memory in use on the fullest GPU, device-wide (NVML, else nvidia-smi); it includes
                              external runtimes such as the SGLang servers
    peak_torch_allocated_gib  torch.cuda.max_memory_allocated in this process, when torch is loaded
    peak_host_rss_gib         this process's peak resident memory (VmHWM)
    peak_host_used_gib        host memory in use (MemTotal - MemAvailable), which counts pinned offload buffers"""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self._handles = _nvml_handles()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu: list[float] = []
        self._host = 0.0

    def _sample(self) -> None:
        for index, used in enumerate(_gpu_used_gib(self._handles)):
            if index >= len(self._gpu):
                self._gpu.append(used)
            else:
                self._gpu[index] = max(self._gpu[index], used)
        host = _host_used_gib()
        if host is not None:
            self._host = max(self._host, host)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self) -> None:
        self._gpu, self._host = [], 0.0
        _reset_peak_rss()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)
        self._sample()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="kuno-bench-memory")
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        allocated = None
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            allocated = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / GIB
        return {
            "peak_gpu_gib": _round(max(self._gpu)) if self._gpu else None,
            "peak_torch_allocated_gib": _round(allocated),
            "peak_host_rss_gib": _round(_peak_rss_gib()),
            "peak_host_used_gib": _round(self._host) if self._host else None,
        }


def attach_step_hooks(pipeline: Any, timer: StepTimer, clock: Clock) -> list[Any]:
    """Times every forward of the denoiser in a loaded pipeline (LtxAdapter's pipelines, H3Adapter's pipeline).
    Returns the hook handles; none when torch is absent or no `transformer` module is found."""
    torch = sys.modules.get("torch")
    if torch is None:
        return []
    holders = [pipeline, *getattr(pipeline, "pipelines", {}).values(), getattr(pipeline, "pipeline", None)]
    modules: dict[int, Any] = {}
    for holder in holders:
        if holder is None:
            continue
        module = getattr(holder, "transformer", None)
        components = getattr(holder, "components", None)
        if module is None and isinstance(components, dict):
            module = components.get("transformer")
        if isinstance(module, torch.nn.Module):
            modules[id(module)] = module
    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    handles = []
    for module in modules.values():
        started: dict[str, float] = {}

        def before(_module, _args, started=started):
            synchronize()
            started["t"] = clock.now()

        def after(_module, _args, _output, started=started):
            synchronize()
            now = clock.now()
            timer.record(now - started.pop("t", now))

        handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
    return handles


# ------------------------------------------------------------------ simulation (--backend mock)

SIMULATED_HOST_RAM_GIB = 128.0
# (cold, warm) load seconds and weights GiB per family; plausible magnitudes, not measurements.
SIMULATED_LOAD_S = {FAMILY_LTX: (95.0, 35.0), FAMILY_H3: (240.0, 80.0)}
SIMULATED_WEIGHTS_GIB = {FAMILY_LTX: 66.0, FAMILY_H3: 124.0}


class SimulatedMemory:
    """MemoryMonitor's interface over the numbers simulated pipelines report."""

    def __init__(self) -> None:
        self.resident_gpu = 0.0
        self.resident_host = 0.0
        self._gpu = 0.0
        self._host = 0.0

    def note(self, gpu_gib: float, host_gib: float) -> None:
        self._gpu = max(self._gpu, gpu_gib)
        self._host = max(self._host, host_gib)

    def start(self) -> None:
        self._gpu, self._host = self.resident_gpu, self.resident_host

    def stop(self) -> dict[str, float | None]:
        return {
            "peak_gpu_gib": _round(self._gpu),
            "peak_torch_allocated_gib": _round(self._gpu * 0.95),
            "peak_host_rss_gib": _round(self._host),
            "peak_host_used_gib": _round(self._host + 4.0),
        }


class SimulatedPipeline:
    """Stands in for a loaded pipeline: advances the clock by a GPU time that grows with the call's latent size,
    records each step, reports stages to the verified-mode tap like the diffusers hooks, and returns a few tiny frames."""

    def __init__(self, profile: ModelProfile, clock: Clock, timer: StepTimer, memory: SimulatedMemory, speed: float = 1.0):
        self.profile = profile
        self.clock = clock
        self.timer = timer
        self.memory = memory
        # Sequence parallelism splits a step across the worker's GPUs, sub-linearly.
        self.speed = speed * profile.gpus_per_worker**0.8

    def __call__(self, **call: Any) -> dict[str, Any]:
        from .backends.quantized import latent_tokens

        frames = call.get("num_frames") or ltx_num_frames(float(call.get("audio_max_duration", 5.0)), round(float(call.get("frame_rate", 24))))
        tokens = latent_tokens(int(call["width"]), int(call["height"]), int(frames))
        stages = self.profile.verified.stage_steps if self.profile.verified else [self.profile.steps]
        tap = call.get("kuno_trajectory_tap")
        rng = np.random.default_rng(int(call["seed"]))
        self.clock.advance(1.0)  # text encoding
        for stage, steps in enumerate(stages):
            stage_tokens = tokens / 4 if len(stages) > 1 and stage == 0 else tokens  # a first stage at half resolution
            per_step = (0.05 + stage_tokens * 1.6e-4 * (1 + stage_tokens / 200_000)) / self.speed
            video = rng.standard_normal((1, 16, 8), dtype=np.float32)
            if tap is not None:
                tap.begin_stage([1.0 - i / steps for i in range(steps)] + [0.0], {"video": video})
            for step in range(steps):
                self.clock.advance(per_step)
                self.timer.record(per_step)
                if tap is not None:
                    video = video * np.float32(0.5)
                    tap.end_step(step, {"video": video})
        self.clock.advance(tokens * 2e-4 / self.speed)  # decoding
        weights = SIMULATED_WEIGHTS_GIB[self.profile.family] / self.profile.gpus_per_worker
        self.memory.note(gpu_gib=weights + tokens * 4e-4, host_gib=16.0 + tokens * 2e-5)
        return {"videos": [[np.full((16, 16, 3), i * 32, dtype=np.uint8) for i in range(4)]], "audio": None, "sampling_rate": 48000}

    def unload(self) -> None:
        self.memory.resident_gpu = self.memory.resident_host = 0.0


def simulated_backends(workdir: Path, clock: Clock, timer: StepTimer, memory: SimulatedMemory, *, hardware_class: str | None, speed: float):
    """The resident LTX-2.5 and H3 backends the worker serves with, over simulated pipelines."""
    from .backends.h3_resident import H3ResidentBackend
    from .backends.ltx_resident import LtxResidentBackend

    loaded: set[str] = set()

    def load(profile: ModelProfile) -> SimulatedPipeline:
        cold, warm = SIMULATED_LOAD_S[profile.family]
        clock.advance((warm if profile.id in loaded else cold) / speed)
        loaded.add(profile.id)
        memory.resident_gpu = SIMULATED_WEIGHTS_GIB[profile.family] / profile.gpus_per_worker
        memory.resident_host = 12.0
        memory.note(memory.resident_gpu, SIMULATED_WEIGHTS_GIB[profile.family] * 0.3)
        return SimulatedPipeline(profile, clock, timer, memory, speed)

    ltx = LtxResidentBackend(None, workdir, loader=load, hardware_class=hardware_class, host_ram_gib=SIMULATED_HOST_RAM_GIB)
    h3 = H3ResidentBackend(workdir, loader=load, hardware_class=hardware_class)
    for backend in (ltx, h3):
        backend._determinism = {"simulated": True}  # verified mode without torch's process pins, which need torch
    return {FAMILY_LTX: ltx, FAMILY_H3: h3}
