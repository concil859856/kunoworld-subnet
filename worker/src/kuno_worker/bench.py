"""kuno-bench: measure what pricing needs on a GPU machine, through the backends jobs use.

    kuno-bench --models-dir /models/ltx-2.5 --profiles ltx-2.5-fast,ltx-2.5-pro --repeats 2 --time-budget 3h --out bench-h200.json
    kuno-bench --backend mock --profiles ltx-2.5-fast --out bench-mock.json      # CPU, simulated timings, same JSON

Per profile the machine can load (default: what `kuno-preflight` says fits, without a TEE, plus the profiles
--hardware-class serves):
  * cold and warm load time: the first load in this process, then a reload after unloading (page cache warm);
  * seconds per denoising step: every forward of the denoiser, timed by hooks on the loaded pipeline;
  * wall time for resolution × fps × duration: each resolution's 16:9 size (or its first), the profile's default
    fps and every fps of 48 or more it allows, and its minimum, 5 s and maximum duration at that fps
    (`limits.max_duration_s_by_fps` caps it);
  * peak GPU memory and host RAM per run (bench_probes.MemoryMonitor);
  * the machine: GPU model, count, driver, and confidential-computing mode from `nvidia-smi conf-compute -f`.

Same code path. Backends come from `build_backends("real", WorkerConfig.from_env())` with the flags below
applied, and each run is `Backend.generate` on a task built like a job's: build_call, admission, verified-mode
commitments when --hardware-class selects them, the resident model store and MP4 encoding. Text-to-video
(reference-to-video with a generated image for h3-reference). Outputs are discarded.

Fixed inputs. Repeat r uses prompt r of a neutral set that passes the shared content policy, and seed --seed + r.
--warmup runs the cheapest cell first and discards it (CUDA kernels, allocator).

Time budget. Cells run cheapest first (pixels × frames). Each profile gets an equal share of the budget left when
it starts; a cell whose predicted time (the largest measured cell scaled by work^1.2) no longer fits is recorded
as skipped with its prediction. Results are rewritten to --out after every profile, so a lost rental keeps them.

Output: a JSON document (`"schema": "kuno-bench"`, `"schema_version": 1`) and a summary table. Feed it to
`kuno-devkit derive-rates`. --backend real needs the worker's `gpu` extra: `uv sync --extra gpu`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kuno_protocol.content_policy import ContentPolicyViolation, check_prompt
from kuno_protocol.profiles import FAMILY_LTX, InputRole, Mode, ModelProfile, ParamError, load_profiles, validate_params

from .backends.base import Backend, GenerationTask
from .bench_probes import Clock, MemoryMonitor, SimulatedMemory, StepTimer, attach_step_hooks, probe_machine, simulated_backends, simulated_machine
from .plan import build_task, example_task

log = logging.getLogger("kuno.bench")

SCHEMA = "kuno-bench"
SCHEMA_VERSION = 1
PROMPTS = (
    "A lighthouse keeper lights the lamp at dusk while waves roll against the rocks",
    "A paper boat drifts down a rain-soaked street past fallen autumn leaves",
    "Time-lapse of clouds drifting over a mountain valley, wind moving through the pine trees",
)
TYPICAL_DURATION_S = 5.0
HIGH_FPS = 48
# Predicted time grows faster than pixels × frames (attention); erring high keeps the budget.
WORK_EXPONENT = 1.2


@dataclass(frozen=True)
class Cell:
    resolution: str
    aspect_ratio: str
    duration_s: float
    fps: int

    @property
    def key(self) -> str:
        return f"{self.resolution}:{self.aspect_ratio}:{self.duration_s:g}:{self.fps}"


def default_cells(profile: ModelProfile) -> list[Cell]:
    lim = profile.limits
    fps_values = sorted({lim.default_fps, *(fps for fps in lim.fps if fps >= HIGH_FPS)})
    cells = []
    for resolution, ratios in lim.sizes.items():
        aspect = "16:9" if "16:9" in ratios else next(iter(ratios))
        for fps in fps_values:
            longest = min(lim.max_duration_s, lim.max_duration_s_by_fps.get(fps, lim.max_duration_s))
            typical = min(max(TYPICAL_DURATION_S, lim.min_duration_s), longest)
            cells += [Cell(resolution, aspect, seconds, fps) for seconds in sorted({lim.min_duration_s, typical, longest})]
    return cells


def parse_cells(spec: str, profile: ModelProfile) -> list[Cell]:
    """`resolution:aspect:seconds[:fps]`, comma-separated; the ones this profile cannot take are dropped."""
    cells = []
    for item in filter(None, (part.strip() for part in spec.split(","))):
        parts = item.split(":")
        if len(parts) not in (4, 5):
            raise SystemExit(f"--cells: {item!r} is not resolution:aspect:seconds[:fps]")
        fps = int(parts[4]) if len(parts) == 5 else profile.limits.default_fps
        cells.append(Cell(parts[0], f"{parts[1]}:{parts[2]}", float(parts[3]), fps))
    return cells


def bench_mode(profile: ModelProfile) -> Mode:
    """A mode that renders one clip: text-to-video, else reference-to-video, else the first other rendering mode. Plans
    render nothing and storyboards chain shots, so neither measures a cell."""
    if Mode.TEXT_TO_VIDEO in profile.modes:
        return Mode.TEXT_TO_VIDEO
    if Mode.REFERENCE_TO_VIDEO in profile.modes:
        return Mode.REFERENCE_TO_VIDEO
    return next((mode for mode in profile.modes if mode not in (Mode.PLAN, Mode.STORYBOARD)), profile.modes[0])


def fits(profile: ModelProfile, mode: Mode, cell: Cell) -> bool:
    if cell.aspect_ratio not in profile.limits.sizes.get(cell.resolution, {}):
        return False
    params = example_task(profile, mode, duration_s=cell.duration_s, resolution=cell.resolution, aspect_ratio=cell.aspect_ratio, fps=cell.fps)
    try:
        validate_params(profile, params)
    except ParamError:
        return False
    return True


def work(profile: ModelProfile, cell: Cell) -> float:
    width, height = profile.size_for(cell.resolution, cell.aspect_ratio)
    return float(width * height * profile.num_frames(cell.duration_s, cell.fps))


def predict_wall(done: list[tuple[float, float]], cell_work: float) -> float | None:
    if not done:
        return None
    ref_work, ref_wall = max(done)
    return ref_wall * (cell_work / ref_work) ** WORK_EXPONENT


def neutral_png(width: int, height: int) -> bytes:
    """A plain sky-to-sea gradient, for modes that need an image."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = bytearray()
    for y in range(height):
        shade = 60 + 140 * y // max(1, height - 1)
        rows += b"\x00" + bytes((shade, shade, 230 - shade // 2)) * width
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b"")


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"n": len(values), "mean": round(statistics.fmean(values), 3), "median": round(statistics.median(values), 3),
            "min": round(min(values), 3), "max": round(max(values), 3)}


def _max(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _failure(exc: BaseException) -> str:
    from .backends.quantized import CapacityRefused

    if isinstance(exc, CapacityRefused):
        return "refused"
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
        return "oom"
    return "error"


def _empty_cuda_cache() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ running


@dataclass
class Bench:
    kind: str
    backends: dict[str, Backend]
    clock: Clock
    timer: StepTimer
    memory: Any
    workdir: Path
    prompts: tuple[str, ...]
    seed: int
    repeats: int
    warmup: int
    check_determinism: bool
    drop_caches: bool
    cells_spec: str | None

    def backend_for(self, profile: ModelProfile) -> Backend:
        backend = self.backends.get(profile.family) or self.backends.get("*")
        if backend is None:
            raise SystemExit(f"no backend for {profile.family}")
        return backend

    def task(self, profile: ModelProfile, mode: Mode, cell: Cell, repeat: int) -> GenerationTask:
        params = example_task(profile, mode, duration_s=cell.duration_s, resolution=cell.resolution, aspect_ratio=cell.aspect_ratio, fps=cell.fps)
        task = build_task(profile, params, self.workdir, seed=self.seed + repeat, prompt=self.prompts[repeat % len(self.prompts)],
                          job_id=f"bench-{uuid.uuid4().hex}")
        for item in task.inputs:
            if item.ref.role in (InputRole.REFERENCE_IMAGE, InputRole.FIRST_FRAME, InputRole.LAST_FRAME, InputRole.KEYFRAME):
                item.data = neutral_png(task.width, task.height)
                item.ref = item.ref.model_copy(update={"size": len(item.data), "sha256": hashlib.sha256(item.data).hexdigest()})
        return task

    def run(self, backend: Backend, profile: ModelProfile, mode: Mode, cell: Cell, repeat: int) -> dict[str, Any]:
        task = self.task(profile, mode, cell, repeat)
        marks: dict[str, float] = {}
        record: dict[str, Any] = {"repeat": repeat, "seed": task.seed, "prompt_index": repeat % len(self.prompts)}
        self.timer.reset()
        self.memory.start()
        begin = self.clock.now()
        try:
            result = backend.generate(task, lambda _value, stage: marks.setdefault(stage, self.clock.now()))
        except Exception as exc:  # the bench records failures; its prompts are its own, so messages are safe to keep
            outcome = _failure(exc)
            record |= {"outcome": outcome, "wall_s": round(self.clock.now() - begin, 3), "detail": f"{type(exc).__name__}: {exc}"[:500]}
            record |= self.memory.stop()
            if outcome == "oom":
                _empty_cuda_cache()
            return record
        wall = self.clock.now() - begin
        record |= self.memory.stop()
        denoise = sum(self.timer.durations) if self.timer.durations else None
        encode = marks["encoded"] - marks["encoding"] if "encoded" in marks and "encoding" in marks else None
        record |= {
            "outcome": "ok",
            "wall_s": round(wall, 3),
            "denoise_s": round(denoise, 3) if denoise is not None else None,
            "transformer_calls": len(self.timer.durations),
            "s_per_step": round(denoise / profile.steps, 4) if denoise is not None else None,
            "encode_s": round(encode, 3) if encode is not None else None,
            "frames": result.info.frames,
            "output_bytes": len(result.data),
            "output_sha256": hashlib.sha256(result.data).hexdigest(),
        }
        if result.openings is not None:
            result.openings.discard()
        return record

    # -------------------------------------------------------------- one profile

    def load(self, backend: Backend, profile: ModelProfile) -> tuple[dict[str, Any], Any]:
        """Cold then warm load through `Backend.warm`. Returns the load block and the resident store (None for an external server)."""
        from .backends.resident import ModelStore

        target = backend
        turbo = getattr(backend, "turbo", None)
        if profile.runtime == "lightx2v" and isinstance(getattr(turbo, "store", None), ModelStore):
            target = turbo  # H3SglangBackend hands the Turbo profile to its resident runtime
        store = getattr(target, "store", None)
        if not isinstance(store, ModelStore):
            backend.warm(profile)
            return {"cold_s": None, "warm_s": None, "note": "an external runtime server; its load is not measured here"}, None
        block: dict[str, Any] = {"page_cache": "as found"}
        if self.drop_caches and self.kind == "real":
            block["page_cache"] = "dropped" if _drop_page_cache() else "drop failed (needs root)"
        for phase in ("cold", "warm"):
            if phase == "warm":
                store.unload_all()
            self.memory.start()
            begin = self.clock.now()
            target.warm(profile)
            block[f"{phase}_s"] = round(self.clock.now() - begin, 2)
            peaks = self.memory.stop()
            block[f"{phase}_peak_gpu_gib"] = peaks["peak_gpu_gib"]
            block[f"{phase}_peak_host_rss_gib"] = peaks["peak_host_rss_gib"]
        return block, store

    def profile(self, profile: ModelProfile, deadline: float | None) -> dict[str, Any]:
        backend = self.backend_for(profile)
        mode = bench_mode(profile)
        cells = parse_cells(self.cells_spec, profile) if self.cells_spec else default_cells(profile)
        cells = sorted((c for c in cells if fits(profile, mode, c)), key=lambda c: work(profile, c))
        record: dict[str, Any] = {
            "profile": profile.id, "family": profile.family, "gpus_per_worker": profile.gpus_per_worker, "steps": profile.steps,
            "mode": mode.value, "backend": getattr(backend, "name", type(backend).__name__), "verified_mode": backend.verified_enabled(profile),
            "hardware_class": getattr(backend, "hardware_class", None), "status": "ok", "detail": None, "load": None,
            "warmup_s": [], "cells": [], "memory_fit": None,
        }
        if not cells:
            return record | {"status": "skipped", "detail": "no benchmark cell fits this profile"}
        if deadline is not None and self.clock.now() >= deadline:
            record["cells"] = [self.cell_record(profile, mode, c, [], skipped="time budget") for c in cells]
            return record | {"status": "skipped", "detail": "time budget"}
        try:
            record["load"], store = self.load(backend, profile)
        except Exception as exc:  # weights, precision or import problems: say so and move on to the next profile
            log.error("%s did not load: %s", profile.id, exc)
            return record | {"status": "load_failed", "detail": f"{type(exc).__name__}: {exc}"[:1000]}
        handles: list[Any] = []
        if store is not None and self.kind == "real":
            with store.acquire(profile) as pipeline:
                handles = attach_step_hooks(pipeline, self.timer, self.clock)
                load_plan = getattr(pipeline, "load_plan", None)  # LtxAdapter: the precision recipe and offload it loaded with
                if load_plan is not None:
                    record["load"] |= {"recipe": load_plan.recipe.id, "offload": load_plan.offload}
            record["step_timing"] = "denoiser hooks" if handles else "unavailable"
        else:
            record["step_timing"] = "simulated" if self.kind == "mock" else "unavailable"
        try:
            warmups = [self.run(backend, profile, mode, cells[0], 0) for _ in range(self.warmup)]
            record["warmup_s"] = [w["wall_s"] for w in warmups]
            if warmups and all(w["outcome"] == "error" for w in warmups):
                record["cells"] = [self.cell_record(profile, mode, c, [], skipped="warmup failed") for c in cells]
                return record | {"status": "failed", "detail": warmups[-1]["detail"]}
            done: list[tuple[float, float]] = []
            for cell in cells:
                predicted = predict_wall(done, work(profile, cell))
                if deadline is not None and (self.clock.now() >= deadline or (predicted is not None and self.clock.now() + predicted * self.repeats > deadline)):
                    record["cells"].append(self.cell_record(profile, mode, cell, [], skipped="time budget", predicted=predicted))
                    continue
                runs = []
                for repeat in range(self.repeats):
                    runs.append(self.run(backend, profile, mode, cell, repeat))
                    if runs[-1]["outcome"] != "ok":
                        break
                if self.check_determinism and runs and runs[0]["outcome"] == "ok":
                    rerun = self.run(backend, profile, mode, cell, 0) | {"determinism_rerun": True}
                    runs.append(rerun)
                cell_record = self.cell_record(profile, mode, cell, runs)
                record["cells"].append(cell_record)
                if cell_record["wall_s"]:
                    done.append((work(profile, cell), cell_record["wall_s"]["median"]))
            record["memory_fit"] = memory_fit(backend, profile, record["cells"])
        finally:
            for handle in handles:
                handle.remove()
            if store is not None:
                store.unload_all()
        return record

    def cell_record(self, profile: ModelProfile, mode: Mode, cell: Cell, runs: list[dict[str, Any]], skipped: str | None = None,
                    predicted: float | None = None) -> dict[str, Any]:
        width, height = profile.size_for(cell.resolution, cell.aspect_ratio)
        task = self.task(profile, mode, cell, 0)
        tokens, estimate = None, None
        if profile.family == FAMILY_LTX:
            from .backends.ltx_resident import build_call
            from .backends.quantized import call_frames, call_tokens

            call = build_call(task)
            tokens = call_tokens(call, width, height)
            plan_for = getattr(self.backend_for(profile), "memory_plan", None)
            try:
                plan = plan_for(profile) if plan_for else None
            except Exception:  # a class that cannot serve the profile already failed its load
                plan = None
            # A job peaks at its render or, on ltx-2.5-4k, at its diffusion decode (0 elsewhere), whichever is larger.
            estimate = round(max(plan.estimate_gib(tokens), plan.decode_gib(width, height, call_frames(call))), 2) if plan is not None else None
        ok = [r for r in runs if r["outcome"] == "ok"]
        walls = [r["wall_s"] for r in ok]
        wall = _stats(walls)
        steps = [r["s_per_step"] for r in ok if r.get("s_per_step") is not None]
        record = {
            "resolution": cell.resolution, "aspect_ratio": cell.aspect_ratio, "width": width, "height": height, "fps": cell.fps,
            "duration_s": cell.duration_s, "frames": profile.num_frames(cell.duration_s, cell.fps), "latent_tokens": tokens,
            "estimate_gib": estimate,
            "outcome": "skipped" if skipped else ("ok" if ok else (runs[-1]["outcome"] if runs else "skipped")),
            "detail": skipped or next((r.get("detail") for r in runs if r["outcome"] != "ok"), None),
            "predicted_wall_s": round(predicted, 1) if predicted is not None else None,
            "wall_s": wall,
            "s_per_step": round(statistics.median(steps), 4) if steps else None,
            "gpu_seconds_per_output_second": round(profile.gpus_per_worker * wall["median"] / cell.duration_s, 3) if wall else None,
            "peak_gpu_gib": _max([r.get("peak_gpu_gib") for r in runs]),
            "peak_torch_allocated_gib": _max([r.get("peak_torch_allocated_gib") for r in runs]),
            "peak_host_rss_gib": _max([r.get("peak_host_rss_gib") for r in runs]),
            "peak_host_used_gib": _max([r.get("peak_host_used_gib") for r in runs]),
            "runs": runs,
        }
        if self.check_determinism and len(ok) >= 2:
            record["deterministic"] = len({r["output_sha256"] for r in ok if r["seed"] == ok[0]["seed"]}) == 1
        return record


def _drop_page_cache() -> bool:
    try:
        subprocess.run(["sync"], check=False, timeout=120)
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def memory_fit(backend: Backend, profile: ModelProfile, cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    """For a class with a memory plan (backends/quantized.py): least squares of the measured torch peaks over latent
    tokens, in precision_recipes.json's terms. Review it, then copy it there with "measured": true."""
    plan_for = getattr(backend, "memory_plan", None)
    try:
        plan = plan_for(profile) if plan_for else None
    except Exception:
        return None
    points = [(c["latent_tokens"], r["peak_torch_allocated_gib"]) for c in cells if c.get("latent_tokens")
              for r in c["runs"] if r["outcome"] == "ok" and r.get("peak_torch_allocated_gib") is not None]
    if plan is None or len({tokens for tokens, _ in points}) < 2:
        return None
    import numpy as np

    from .backends.quantized import resolve_recipe

    recipe, _ = resolve_recipe(profile, getattr(backend, "hardware_class", None))
    weights_on_gpu = plan.token_base_gib - recipe.memory.activation_fixed_gib
    x = np.array([[1.0, tokens / 10_000] for tokens, _ in points])
    y = np.array([peak - weights_on_gpu - plan.overhead_gib for _, peak in points])
    (fixed, per_10k), *_ = np.linalg.lstsq(x, y, rcond=None)
    return {"recipe": recipe.id, "offload": plan.offload, "activation_fixed_gib": round(float(fixed), 3),
            "activation_gib_per_10k_tokens": round(float(per_10k), 3), "samples": len(points),
            "note": "peaks are torch.cuda.max_memory_allocated; add allocator slack before publishing"}


# ------------------------------------------------------------------ output


def summary_table(doc: dict[str, Any]) -> str:
    machine = doc["machine"]
    cc = machine.get("cc_mode") or "unknown"
    lines = [
        f"kuno-bench ({doc['backend']}{', SIMULATED' if machine.get('simulated') else ''}): {machine.get('gpu_count')} × {machine.get('gpu_model')}, "
        f"driver {machine.get('driver')}, CC mode {cc}",
    ]
    header = ("profile", "res", "fps", "dur s", "outcome", "wall s", "s/step", "GPU-s/out-s", "peak GPU GiB", "peak RSS GiB")
    rows = [header]
    for profile in doc["profiles"]:
        load = profile.get("load") or {}
        lines.append(f"{profile['profile']}: {profile['status']}" + (f" ({profile['detail']})" if profile.get("detail") else "")
                     + (f", load cold {load.get('cold_s')} s / warm {load.get('warm_s')} s" if load.get("cold_s") is not None else ""))
        for cell in profile["cells"]:
            wall = cell["wall_s"]
            rows.append((profile["profile"], cell["resolution"], str(cell["fps"]), f"{cell['duration_s']:g}", cell["outcome"],
                         f"{wall['median']:.1f}" if wall else (f"~{cell['predicted_wall_s']:.0f}" if cell.get("predicted_wall_s") else "-"),
                         f"{cell['s_per_step']:.3f}" if cell.get("s_per_step") is not None else "-",
                         f"{cell['gpu_seconds_per_output_second']:.2f}" if cell.get("gpu_seconds_per_output_second") is not None else "-",
                         f"{cell['peak_gpu_gib']:.1f}" if cell.get("peak_gpu_gib") is not None else "-",
                         f"{cell['peak_host_rss_gib']:.1f}" if cell.get("peak_host_rss_gib") is not None else "-"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines.append("")
    lines += ["  ".join(value.ljust(widths[i]) for i, value in enumerate(row)).rstrip() for row in rows]
    return "\n".join(lines)


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, path)


def parse_seconds(text: str) -> float:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", text)
    if not match:
        raise argparse.ArgumentTypeError(f"{text!r} is not a duration such as 5400, 90m or 2h")
    return float(match.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]


def loadable_profiles(host: Any, hardware_class: str | None) -> list[str]:
    from .preflight import profile_blockers

    chosen = []
    for profile in load_profiles().values():
        by_class = hardware_class and profile.verified and profile.verified.hardware_class(hardware_class) and len(host.gpus) >= profile.gpus_per_worker
        if not profile_blockers(profile, host, require_tee=False) or by_class:
            chosen.append(profile.id)
    return chosen


def real_backends(args: argparse.Namespace, profile_ids: list[str], workdir: Path) -> dict[str, Backend]:
    from .backends import build_backends
    from .config import WorkerConfig

    config = WorkerConfig.from_env()
    config.workdir = workdir
    overrides = {"ltx_models_dir": args.models_dir, "verified_hardware_class": args.hardware_class, "model_digest": args.model_digest,
                 "ltx_offload": args.offload, "weights_verify": args.weights_verify, "h3_turbo_lora": args.h3_turbo_lora}
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    if args.allow_unpinned:
        config.allow_unpinned_weights = True
    catalog = load_profiles()
    if config.ltx_models_dir is None and not any(catalog[p].family == FAMILY_LTX for p in profile_ids):
        config.ltx_models_dir = workdir / "no-ltx-weights"  # only H3 is benchmarked; the LTX backend is built but never loads
    try:
        return build_backends("real", config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kuno-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("real", "mock"), default="real", help="mock: simulated pipelines on CPU (tests)")
    parser.add_argument("--profiles", help="comma-separated profile ids (default: every profile this machine can load)")
    parser.add_argument("--repeats", type=int, default=2, help="measured runs per cell (default 2)")
    parser.add_argument("--warmup", type=int, default=1, help="discarded runs of the cheapest cell per profile (default 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", action="append", help="replace the neutral prompt set (repeatable; must pass the content policy)")
    parser.add_argument("--time-budget", type=parse_seconds, help="e.g. 5400, 90m or 2h; the slowest cells are skipped to fit")
    parser.add_argument("--cells", help="resolution:aspect:seconds[:fps],... instead of the default matrix")
    parser.add_argument("--check-determinism", action="store_true", help="rerun each cell's first repeat and compare output hashes")
    parser.add_argument("--drop-caches", action="store_true", help="drop the page cache before the cold load (root)")
    parser.add_argument("--out", type=Path, required=True, help="results JSON, rewritten after every profile")
    parser.add_argument("--workdir", type=Path, help="scratch directory for job files (default: a new temporary directory)")
    real = parser.add_argument_group("real backend (defaults from the worker's KUNO_* environment)")
    real.add_argument("--models-dir", type=Path, help="LTX-2.5 weights (KUNO_LTX_MODELS_DIR)")
    real.add_argument("--hardware-class", help="verified hardware class (KUNO_VERIFIED_HARDWARE_CLASS); selects precision and verified mode")
    real.add_argument("--model-digest", help="KUNO_MODEL_DIGEST")
    real.add_argument("--offload", choices=("auto", "none", "model", "group"), help="KUNO_LTX_OFFLOAD")
    real.add_argument("--weights-verify", choices=("full", "size"), help="KUNO_WEIGHTS_VERIFY (full hashes the weights inside the cold load)")
    real.add_argument("--allow-unpinned", action="store_true", help="KUNO_WEIGHTS_ALLOW_UNPINNED=1 (development)")
    real.add_argument("--h3-turbo-lora", help="KUNO_H3_TURBO_LORA")
    mock = parser.add_argument_group("mock backend")
    mock.add_argument("--mock-gpu", default="NVIDIA H200", help="the GPU the simulated machine reports")
    mock.add_argument("--mock-speed", type=float, default=1.0, help="simulated GPU speed; 2 halves every simulated time")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--repeats must be at least 1 and --warmup at least 0")
    prompts = tuple(args.prompt) if args.prompt else PROMPTS
    for prompt in prompts:
        try:
            check_prompt(prompt)
        except ContentPolicyViolation:
            raise SystemExit("a --prompt does not pass the content policy") from None
    catalog = load_profiles()
    requested = [p.strip() for p in (args.profiles or "").split(",") if p.strip()]
    unknown = [p for p in requested if p not in catalog]
    if unknown:
        raise SystemExit(f"unknown profile(s): {', '.join(unknown)}; choose from {', '.join(catalog)}")
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="kuno-bench-"))
    clock, timer = Clock(), StepTimer()
    if args.backend == "mock":
        requested = requested or list(catalog)
        machine = simulated_machine(args.mock_gpu, max(catalog[p].gpus_per_worker for p in requested))
        memory: Any = SimulatedMemory()
        backends = simulated_backends(workdir, clock, timer, memory, hardware_class=args.hardware_class, speed=args.mock_speed)
    else:
        host, machine = probe_machine()
        requested = requested or loadable_profiles(host, args.hardware_class)
        if not requested:
            raise SystemExit("no profile fits this machine (see kuno-preflight --no-tee); pass --profiles to try one anyway")
        memory = MemoryMonitor()
        backends = real_backends(args, requested, workdir)
    bench = Bench(kind=args.backend, backends=backends, clock=clock, timer=timer, memory=memory, workdir=workdir, prompts=prompts,
                  seed=args.seed, repeats=args.repeats, warmup=args.warmup, check_determinism=args.check_determinism,
                  drop_caches=args.drop_caches, cells_spec=args.cells)
    started = clock.now()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": args.backend,
        "simulated": args.backend == "mock",
        "machine": machine,
        "config": {
            "profiles": requested, "repeats": args.repeats, "warmup": args.warmup, "seed": args.seed, "prompts": list(prompts),
            "time_budget_s": args.time_budget, "cells": args.cells, "check_determinism": args.check_determinism,
            "hardware_class": args.hardware_class, "offload": args.offload, "weights_verify": args.weights_verify,
        },
        "profiles": [],
        "elapsed_s": None,
    }
    deadline_at = None if args.time_budget is None else started + args.time_budget
    for index, profile_id in enumerate(requested):
        profile_deadline = None
        if deadline_at is not None:
            profile_deadline = clock.now() + max(0.0, deadline_at - clock.now()) / (len(requested) - index)
        log.info("benchmarking %s", profile_id)
        doc["profiles"].append(bench.profile(catalog[profile_id], profile_deadline))
        doc["elapsed_s"] = round(clock.now() - started, 1)
        _write(args.out, doc)
    print(summary_table(doc))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
