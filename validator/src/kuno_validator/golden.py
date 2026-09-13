"""Golden sets: per-step latent hashes for fixed prompts and seeds, per hardware class.

Before a miner image may earn in verified mode, its trajectories must match the reference
bit for bit on the hardware class it declares. A golden set is computed once per
(profile, runtime, hardware class, pinned image) on reference hardware and published; a
candidate image is checked by running the same cases on that class and comparing every
leaf's latent digest. Leaf digests are unsalted latent hashes, so they compare across jobs
even though each job's Merkle root is salted.

    python -m kuno_validator.golden compute --profile ltx-2.5-fast --hardware-class dev-cpu --out golden.json
    python -m kuno_validator.golden check --golden golden.json --leaves observed.json

`compute` uses the reference executor for the class's runtime (only the toy denoiser ships
executors that run without a GPU). `check` compares an observed run, a JSON object mapping
each case name to its list of leaf latent digests, e.g. dumped from a worker's retention
store or from `include_leaves` audit openings of canaries sent with the golden prompts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from kuno_protocol.canonical import canonical_json, sha256_hex
from kuno_protocol.profiles import Mode, ModelProfile, load_profiles
from kuno_protocol.schemas import GenerationParams
from kuno_protocol.toy_denoiser import TOY_RUNTIME, toy_model_digest, toy_transcript
from kuno_protocol.verified import StepTranscript

from .audits import CanaryRecord, StepExecutor, ToyStepExecutor
from .canaries import FALLBACK_PROMPTS


class GoldenCase(BaseModel):
    name: str
    profile_id: str
    params: GenerationParams
    prompt: str
    negative_prompt: str | None = None
    seed: int


class GoldenEntry(BaseModel):
    case: GoldenCase
    leaf_digests: list[str]


class GoldenSet(BaseModel):
    v: Literal[1] = 1
    profile_id: str
    runtime: str
    hardware_class: str
    model_digest: str
    image_digest: str | None = None
    created_at: float
    entries: list[GoldenEntry]

    def entry(self, name: str) -> GoldenEntry | None:
        return next((e for e in self.entries if e.case.name == name), None)


def default_cases(profile: ModelProfile, count: int = 3) -> list[GoldenCase]:
    resolution = next(iter(profile.limits.sizes))
    aspect = "16:9" if "16:9" in profile.limits.sizes[resolution] else next(iter(profile.limits.sizes[resolution]))
    params = GenerationParams(
        profile_id=profile.id, mode=Mode.TEXT_TO_VIDEO, duration_s=profile.limits.min_duration_s, resolution=resolution,
        aspect_ratio=aspect, fps=profile.limits.default_fps, audio=profile.limits.audio,
    )
    return [
        GoldenCase(name=f"{profile.id}-{i}", profile_id=profile.id, params=params, prompt=FALLBACK_PROMPTS[i % len(FALLBACK_PROMPTS)], seed=1000 + i)
        for i in range(count)
    ]


# A run_case callable returns the leaf latent digests one run of a case produced.
RunCase = Callable[[GoldenCase], list[str]]


def toy_reference_runner(profile: ModelProfile, hardware_class: str, executor: StepExecutor | None = None) -> RunCase:
    executor = executor or ToyStepExecutor()

    def run(case: GoldenCase) -> list[str]:
        transcript = _toy_case_transcript(profile, hardware_class, case)
        canary = CanaryRecord(case.name, profile.id, case.params.model_dump(mode="json"), case.prompt, case.seed, {}, case.negative_prompt)
        return executor.trajectory(transcript, canary)

    return run


def _toy_case_transcript(profile: ModelProfile, hardware_class: str, case: GoldenCase) -> StepTranscript:
    return toy_transcript(
        job_id=f"golden-{case.name}",
        params_digest=sha256_hex(canonical_json(case.params.model_dump(mode="json"))),
        profile_id=profile.id, family=profile.family, model_digest=toy_model_digest(profile.id, profile.checkpoint),
        seed=case.seed, prompt=case.prompt, negative_prompt=case.negative_prompt,
        frames=profile.num_frames(case.params.duration_s, case.params.fps), steps=profile.steps, hardware_class=hardware_class,
    )


def worker_backend_runner(backend, profile: ModelProfile) -> RunCase:
    """Runs a case through a worker backend (the candidate image's code) and reads its committed leaves.

    Imports the worker package lazily: only a workspace that has both installed can use it.
    """
    from kuno_worker.backends.base import GenerationTask

    def run(case: GoldenCase) -> list[str]:
        width, height = profile.size_for(case.params.resolution, case.params.aspect_ratio)
        job_id = f"golden-{case.name}"
        task = GenerationTask(
            job_id=job_id, profile=profile, params=case.params, prompt=case.prompt, negative_prompt=case.negative_prompt,
            seed=case.seed, width=width, height=height,
        )
        result = backend.generate(task, lambda _value, _stage: None)
        if result.openings is None:
            raise RuntimeError(f"{backend.name} did not run {profile.id} in verified mode")
        record = result.openings.store.record(job_id)
        try:
            return [leaf.latent for leaf in record.leaves]
        finally:
            result.openings.discard()

    return run


def compute_golden(
    profile: ModelProfile, hardware_class: str, run_case: RunCase, cases: list[GoldenCase] | None = None,
    runtime: str | None = None, model_digest: str | None = None, image_digest: str | None = None,
) -> GoldenSet:
    if profile.verified is None or profile.verified.hardware_class(hardware_class) is None:
        raise ValueError(f"{profile.id} has no verified mode on hardware class {hardware_class}")
    hardware = profile.verified.hardware_class(hardware_class)
    cases = cases or default_cases(profile)
    return GoldenSet(
        profile_id=profile.id,
        runtime=runtime or (TOY_RUNTIME if hardware.dev else profile.verified.runtime),
        hardware_class=hardware_class,
        model_digest=model_digest or (toy_model_digest(profile.id, profile.checkpoint) if hardware.dev else ""),
        image_digest=image_digest,
        created_at=time.time(),
        entries=[GoldenEntry(case=case, leaf_digests=run_case(case)) for case in cases],
    )


def compare_leaves(entry: GoldenEntry, observed: list[str]) -> str | None:
    if len(observed) != len(entry.leaf_digests):
        return f"{entry.case.name}: {len(observed)} leaves, golden has {len(entry.leaf_digests)} (steps skipped or added)"
    for index, (want, got) in enumerate(zip(entry.leaf_digests, observed)):
        if want != got:
            return f"{entry.case.name}: diverges at leaf {index}" + (" (initial noise)" if index == 0 else "")
    return None


@dataclass
class GoldenReport:
    ok: bool
    checked: int
    mismatches: list[str] = field(default_factory=list)


def check_observed(golden: GoldenSet, observed: dict[str, list[str]]) -> GoldenReport:
    mismatches = []
    for entry in golden.entries:
        leaves = observed.get(entry.case.name)
        if leaves is None:
            mismatches.append(f"{entry.case.name}: not run")
            continue
        reason = compare_leaves(entry, leaves)
        if reason:
            mismatches.append(reason)
    return GoldenReport(ok=not mismatches, checked=len(golden.entries), mismatches=mismatches)


def check_image(golden: GoldenSet, run_case: RunCase) -> GoldenReport:
    """Runs every golden case through the candidate and compares all leaves."""
    return check_observed(golden, {entry.case.name: run_case(entry.case) for entry in golden.entries})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kuno_validator.golden", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    compute = sub.add_parser("compute", help="compute a golden set with the reference executor")
    compute.add_argument("--profile", required=True)
    compute.add_argument("--hardware-class", required=True)
    compute.add_argument("--image-digest")
    compute.add_argument("--cases", type=int, default=3)
    compute.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("check", help="compare observed leaf digests with a golden set")
    check.add_argument("--golden", type=Path, required=True)
    check.add_argument("--leaves", type=Path, required=True, help='JSON: {"case name": ["leaf digest", ...]}')
    args = parser.parse_args(argv)

    if args.command == "compute":
        profile = load_profiles()[args.profile]
        hardware = profile.verified.hardware_class(args.hardware_class) if profile.verified else None
        if hardware is None or not hardware.dev:
            print("only simulated hardware classes have a reference executor that runs without a GPU", file=sys.stderr)
            return 2
        golden = compute_golden(
            profile, args.hardware_class, toy_reference_runner(profile, args.hardware_class),
            default_cases(profile, args.cases), image_digest=args.image_digest,
        )
        args.out.write_text(golden.model_dump_json(indent=2))
        print(f"wrote {len(golden.entries)} golden case(s) for {profile.id} on {args.hardware_class} to {args.out}")
        return 0

    golden = GoldenSet.model_validate_json(args.golden.read_text())
    report = check_observed(golden, json.loads(args.leaves.read_text()))
    for line in report.mismatches:
        print(line)
    print(f"{'PASS' if report.ok else 'FAIL'}: {report.checked} case(s), {len(report.mismatches)} mismatch(es)")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
