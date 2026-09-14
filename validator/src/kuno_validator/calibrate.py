"""Calibrating tolerance-mode step audits (VERIFIED_MODE.md, "Tolerance mode").

The owner fills `kuno_protocol/tolerance_calibration.json` (or a validator's
KUNO_TOLERANCE_CALIBRATION file) from real GPU runs. Per (profile, miner hardware class,
executor class):

1. On the miner class, generate golden-case jobs in verified mode and keep their openings.
2. On the executor class, replay every replayable step of them and write one sample per step:
   `StepDistance.record(step)` -> {"step", "rel_l2", "max_abs_rel"} as JSON Lines (honest.jsonl).
3. Optionally do the same for deliberate substitutions (another checkpoint, a coarser quantization,
   a skipped step) into substituted.jsonl.
4. `python -m kuno_validator.calibrate summarize --profile P --hardware-class C --executor-class E
       --honest honest.jsonl [--substituted substituted.jsonl] --calibration tolerance_calibration.json`
   computes the distributions, proposes a threshold (margin × worst honest distance, and below the
   best substitution) and writes the entry. It refuses when there are too few samples or the two
   distributions overlap.

`toy` produces both sample files from the toy denoiser with injected float noise, so the procedure
can be rehearsed without a GPU. Its numbers say nothing about real hardware.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

from kuno_protocol.tolerance import (
    ANY_EXECUTOR,
    Calibration,
    CalibrationEntry,
    CalibrationError,
    load_calibration,
    propose_threshold,
    step_distance,
    summarize,
)


def read_samples(path: Path) -> list[dict]:
    samples = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            samples.append({"step": int(item["step"]), "rel_l2": float(item["rel_l2"]), "max_abs_rel": float(item.get("max_abs_rel", 0.0))})
        except (ValueError, KeyError, TypeError):
            raise CalibrationError(f"{path}:{number} is not a sample line") from None
    return samples


def build_entry(
    profile_id: str,
    hardware_class: str,
    honest: list[dict],
    substituted: list[dict] | None = None,
    *,
    executor_class: str = ANY_EXECUTOR,
    margin: float = 2.0,
    min_samples: int = 200,
    max_abs: bool = False,
    image_digest: str | None = None,
    runtime: str | None = None,
    notes: str | None = None,
) -> CalibrationEntry:
    honest_l2 = [s["rel_l2"] for s in honest]
    cheat_l2 = [s["rel_l2"] for s in substituted] if substituted else None
    threshold = propose_threshold(honest_l2, cheat_l2, margin=margin, min_samples=min_samples)
    max_abs_threshold = None
    if max_abs:
        cheat_abs = [s["max_abs_rel"] for s in substituted] if substituted else None
        max_abs_threshold = propose_threshold([s["max_abs_rel"] for s in honest], cheat_abs, margin=margin, min_samples=min_samples)
    return CalibrationEntry(
        profile_id=profile_id, hardware_class=hardware_class, executor_class=executor_class,
        honest=summarize(honest_l2), honest_max_abs=summarize(s["max_abs_rel"] for s in honest),
        substituted=summarize(cheat_l2) if cheat_l2 else None, threshold=threshold, max_abs_threshold=max_abs_threshold,
        image_digest=image_digest, runtime=runtime, calibrated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )


def toy_samples(count: int, noise: float, seed: int = 1, substitute: bool = False) -> list[dict]:
    """Replays of toy-denoiser steps whose commitments carry `noise` relative float error per step (honest drift),
    or, with `substitute`, whose step was computed with other weights."""
    import numpy as np

    from kuno_protocol.toy_denoiser import toy_conditioning, toy_latent_spec, toy_noise, toy_sigmas, toy_state, toy_step, toy_weights

    rng = random.Random(seed)
    generator = np.random.default_rng(seed)
    samples = []
    spec = toy_latent_spec(49)
    honest, cheap = toy_weights("a" * 64), toy_weights("f" * 64)
    while len(samples) < count:
        job_seed, steps = rng.randrange(2**31), 11
        sigmas = toy_sigmas(steps)
        cond = toy_conditioning(f"calibration prompt {job_seed}")
        x = toy_noise(job_seed, spec)
        for k in range(1, steps + 1):
            replay = toy_step(x, sigmas[k - 1], sigmas[k], cond, honest)
            committed = toy_step(x, sigmas[k - 1], sigmas[k], cond, cheap if substitute else honest)
            update = committed - x
            drift = generator.standard_normal(committed.shape).astype(np.float32) * np.float32(noise) * np.float32(np.sqrt(np.mean(update**2)))
            committed = (committed + drift).astype(np.float32)
            samples.append(step_distance(toy_state(x), toy_state(committed), toy_state(replay)).record(k))
            x = committed
            if len(samples) >= count:
                break
    return samples


def _write_samples(path: Path, samples: list[dict]) -> None:
    path.write_text("".join(json.dumps(s) + "\n" for s in samples))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kuno_validator.calibrate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    summarize_cmd = sub.add_parser("summarize", help="turn measured samples into a calibration entry")
    summarize_cmd.add_argument("--profile", required=True)
    summarize_cmd.add_argument("--hardware-class", required=True, help="the miner's hardware class")
    summarize_cmd.add_argument("--executor-class", default=ANY_EXECUTOR, help="the class the validator replays on (default: any)")
    summarize_cmd.add_argument("--honest", type=Path, required=True)
    summarize_cmd.add_argument("--substituted", type=Path)
    summarize_cmd.add_argument("--margin", type=float, default=2.0)
    summarize_cmd.add_argument("--min-samples", type=int, default=200)
    summarize_cmd.add_argument("--max-abs", action="store_true", help="also set a max_abs_rel threshold")
    summarize_cmd.add_argument("--image-digest")
    summarize_cmd.add_argument("--runtime")
    summarize_cmd.add_argument("--notes")
    summarize_cmd.add_argument("--calibration", type=Path, required=True, help="calibration file to update (created if missing)")
    toy_cmd = sub.add_parser("toy", help="rehearse with the toy denoiser: write honest and substituted sample files")
    toy_cmd.add_argument("--samples", type=int, default=300)
    toy_cmd.add_argument("--noise", type=float, default=1e-4, help="honest drift, relative to each step's update")
    toy_cmd.add_argument("--out-dir", type=Path, required=True)
    show_cmd = sub.add_parser("show", help="print a calibration file (default: the packaged one)")
    show_cmd.add_argument("--calibration", type=Path)
    args = parser.parse_args(argv)

    if args.command == "toy":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _write_samples(args.out_dir / "honest.jsonl", toy_samples(args.samples, args.noise))
        _write_samples(args.out_dir / "substituted.jsonl", toy_samples(args.samples, args.noise, seed=2, substitute=True))
        print(f"Wrote {args.out_dir / 'honest.jsonl'} and {args.out_dir / 'substituted.jsonl'}")
        return 0
    if args.command == "show":
        print(load_calibration(args.calibration).model_dump_json(indent=2))
        return 0
    try:
        entry = build_entry(
            args.profile, args.hardware_class, read_samples(args.honest), read_samples(args.substituted) if args.substituted else None,
            executor_class=args.executor_class, margin=args.margin, min_samples=args.min_samples, max_abs=args.max_abs,
            image_digest=args.image_digest, runtime=args.runtime, notes=args.notes,
        )
    except CalibrationError as exc:
        parser.exit(2, f"calibration refused: {exc}\n")
    calibration = load_calibration(args.calibration) if args.calibration.exists() else Calibration()
    args.calibration.write_text(calibration.with_entry(entry).model_dump_json(indent=2) + "\n")
    print(f"{entry.profile_id} on {entry.hardware_class} (executor {entry.executor_class}): threshold {entry.threshold:.4g} "
          f"from {entry.honest.samples} honest samples (worst {entry.honest.max:.4g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
