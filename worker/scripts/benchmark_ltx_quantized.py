"""Benchmark an LTX-2.5 hardware class on the GPU it declares: load time, seconds per job and per step,
peak VRAM against the recipe's estimate, refusals and out-of-memory errors, for a matrix of request sizes.

For the owner, on the real card. Nothing here runs in CI, and no number in this repository comes from it
yet: until it has run, MINING.md lists every consumer-card speed as unmeasured.

    uv run python subnet/worker/scripts/benchmark_ltx_quantized.py \
        --models-dir /models/ltx-2.5 --hardware-class O1.rtx-5090-32gb.x1.fp8-cast --model-digest <manifest digest> \
        --requests 720p:16:9:2,720p:16:9:5,720p:16:9:10,1080p:16:9:5 --repeat 2 --out bench-5090.jsonl

Options: --offload auto|none|model|group, --profile (default ltx-2.5-fast), --allow-unpinned (development),
--check-determinism (runs every request twice with one seed in verified mode and compares output hashes).

Each line of --out is one JSON record: an `environment` record, one `run` per request, and a final
`memory_fit` with the activation_fixed_gib and activation_gib_per_10k_tokens a least-squares fit gives
over the measured peaks. Review it, then copy it into precision_recipes.json with "measured": true.
Needs the GPU image's torch, diffusers (and torchao for int8) and ffmpeg.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from importlib import metadata
from pathlib import Path


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--hardware-class", required=True)
    parser.add_argument("--profile", default="ltx-2.5-fast")
    parser.add_argument("--model-digest")
    parser.add_argument("--allow-unpinned", action="store_true")
    parser.add_argument("--offload", default="auto")
    parser.add_argument("--requests", default="720p:16:9:2,720p:16:9:5,720p:16:9:10,1080p:16:9:5", help="resolution:aspect:seconds[:fps], comma-separated")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch

    from kuno_protocol.profiles import Mode, load_profiles
    from kuno_worker.backends.ltx_resident import LtxResidentBackend, build_call
    from kuno_worker.backends.quantized import CapacityRefused, call_tokens, host_memory_gib, probe_device
    from kuno_worker.plan import build_task, example_task

    profile = load_profiles()[args.profile]
    workdir = Path(tempfile.mkdtemp(prefix="kuno-bench-"))
    backend = LtxResidentBackend(
        args.models_dir, workdir, hardware_class=args.hardware_class, model_digest=args.model_digest, offload=args.offload,
        allow_unpinned_weights=args.allow_unpinned,
    )
    records: list[dict] = []

    def emit(record: dict) -> None:
        records.append(record)
        with args.out.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))

    device = probe_device()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    backend.warm(profile)
    loaded = next(iter(backend.store._pipelines.values()))
    plan = backend.memory_plan(profile)
    load_plan = getattr(loaded, "load_plan", None)
    emit({
        "record": "environment", "profile": profile.id, "hardware_class": args.hardware_class, "gpu": device.name,
        "gpu_total_gib": round(device.total_gib, 2), "capability": device.capability, "host_ram_gib": host_memory_gib(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "diffusers": _version("diffusers"), "torchao": _version("torchao"),
        "transformers": _version("transformers"), "recipe": load_plan.recipe.id if load_plan else None,
        "offload": load_plan.offload if load_plan else args.offload, "weights_digest": load_plan.weights.model_digest if load_plan else None,
        "load_s": round(time.time() - started, 1), "load_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    })

    for spec in args.requests.split(","):
        parts = spec.split(":")
        resolution, aspect, seconds = parts[0], f"{parts[1]}:{parts[2]}", float(parts[3])
        fps = int(parts[4]) if len(parts) > 4 else profile.limits.default_fps
        params = example_task(profile, Mode.TEXT_TO_VIDEO, resolution=resolution, aspect_ratio=aspect, duration_s=seconds, fps=fps, audio=True)
        for attempt in range(max(args.repeat, 2 if args.check_determinism else 1)):
            task = build_task(profile, params, workdir, seed=args.seed, job_id=f"bench-{resolution}-{seconds:g}-{attempt}")
            tokens = call_tokens(build_call(task), task.width, task.height)
            marks: dict[str, float] = {}
            record = {"record": "run", "request": spec, "attempt": attempt, "latent_tokens": tokens,
                      "estimate_gib": round(plan.estimate_gib(tokens), 2) if plan else None, "steps": profile.steps}
            torch.cuda.reset_peak_memory_stats()
            begin = time.time()
            try:
                result = backend.generate(task, lambda value, stage: marks.setdefault(stage, time.time()))
            except CapacityRefused as exc:
                emit(record | {"outcome": "refused", "detail": str(exc)})
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                emit(record | {"outcome": "oom", "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)})
                break
            denoise = marks.get("encoding", time.time()) - marks.get("denoising", begin)
            emit(record | {
                "outcome": "ok", "wall_s": round(time.time() - begin, 2), "denoise_s": round(denoise, 2),
                "s_per_step": round(denoise / profile.steps, 3),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                "output_sha256": hashlib.sha256(result.data).hexdigest(), "committed": result.step_commitment is not None,
            })

    if args.check_determinism:
        by_request: dict[str, set[str]] = {}
        for r in records:
            if r.get("outcome") == "ok":
                by_request.setdefault(r["request"], set()).add(r["output_sha256"])
        emit({"record": "determinism", "identical_outputs": {k: len(v) == 1 for k, v in by_request.items()}})

    runs = [r for r in records if r.get("outcome") == "ok"]
    if plan is not None and len({r["latent_tokens"] for r in runs}) >= 2:
        weights_on_gpu = plan.token_base_gib - load_plan.recipe.memory.activation_fixed_gib if load_plan else 0.0
        x = np.array([[1.0, r["latent_tokens"] / 10_000] for r in runs])
        y = np.array([r["peak_allocated_gib"] - weights_on_gpu - plan.overhead_gib for r in runs])
        (fixed, per_10k), *_ = np.linalg.lstsq(x, y, rcond=None)
        emit({"record": "memory_fit", "offload": plan.offload, "activation_fixed_gib": round(float(fixed), 3),
              "activation_gib_per_10k_tokens": round(float(per_10k), 3), "samples": len(runs),
              "note": "peaks are torch.cuda.max_memory_allocated; add allocator slack before publishing"})


if __name__ == "__main__":
    main()
