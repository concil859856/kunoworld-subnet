"""kuno-verified-check: does this hardware class commit the same trajectory twice? (VERIFIED_MODE.md, "Phase 0")

    kuno-verified-check run --profile ltx-2.5-fast --hardware-class C1.rtx-pro-6000-bw-se.x1 \
        --models-dir /models/ltx-2.5 --out run-a.json
    kuno-verified-check run --cases run-a.json --models-dir /models/ltx-2.5 --out run-b.json   # a second process
    kuno-verified-check compare run-a.json run-b.json

Phase 0 step 1, before a GPU class may earn in verified mode: the same cases, rendered twice in two processes,
must commit the same per-step latents. `run` renders each case through the backend the worker serves with, in
verified mode, and writes the leaf digests the receipt's Merkle tree is built from. Leaves are unsalted latent
hashes, so they compare across jobs, processes and machines.

`compare` names the first step where two runs differ, which says what diverged: the conditioning digest is the
text encoder, leaf 0 is the seed's noise, a later leaf is the denoiser. It also compares the transcripts, so a
run against different weights or a different class fails loudly rather than quietly.

A run file carries the cases it ran, so the second process (or the second machine) takes `--cases run-a.json` and
cannot drift, and `python -m kuno_validator.golden adopt` turns a run into the published golden set (step 3).

--backend mock runs the simulated pipelines on CPU, which exercises this tool and proves nothing about a GPU;
--backend real needs the worker's `gpu` extra (`uv sync --extra gpu`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kuno_protocol.content_policy import ContentPolicyViolation, check_prompt
from kuno_protocol.profiles import InputRole, Mode, ModelProfile, load_profiles
from kuno_protocol.prompts import NEUTRAL_PROMPTS
from kuno_protocol.schemas import GenerationParams

from .backends.base import Backend
from .plan import build_task, example_task

log = logging.getLogger("kuno.verified-check")

SCHEMA = "kuno-verified-check"
SCHEMA_VERSION = 1
DEFAULT_CASES = 3
FIRST_SEED = 1000  # kuno_validator.golden.default_cases seeds the same way, so both build the same cases


class VerifiedCheckError(RuntimeError):
    """The check cannot be run or the two runs are not comparable (as opposed to a divergence, which is a result)."""


@dataclass(frozen=True)
class Case:
    """One golden case. The same fields as kuno_validator.golden.GoldenCase, which reads them straight back."""

    name: str
    profile_id: str
    params: GenerationParams
    prompt: str
    negative_prompt: str | None
    seed: int

    def as_json(self) -> dict[str, Any]:
        return {"name": self.name, "profile_id": self.profile_id, "params": self.params.model_dump(mode="json"),
                "prompt": self.prompt, "negative_prompt": self.negative_prompt, "seed": self.seed}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Case:
        return cls(name=data["name"], profile_id=data["profile_id"], params=GenerationParams.model_validate(data["params"]),
                   prompt=data["prompt"], negative_prompt=data.get("negative_prompt"), seed=int(data["seed"]))


def default_cases(profile: ModelProfile, count: int = DEFAULT_CASES, *, duration_s: float | None = None,
                  resolution: str | None = None, aspect_ratio: str | None = None, fps: int | None = None) -> list[Case]:
    """What kuno_validator.golden.default_cases builds: the shortest clip at the first resolution, neutral prompts."""
    mode = Mode.TEXT_TO_VIDEO if Mode.TEXT_TO_VIDEO in profile.modes else next(iter(profile.modes))
    params = example_task(profile, mode, duration_s=duration_s, resolution=resolution, aspect_ratio=aspect_ratio, fps=fps)
    return [
        Case(name=f"{profile.id}-{i}", profile_id=profile.id, params=params, prompt=NEUTRAL_PROMPTS[i % len(NEUTRAL_PROMPTS)],
             negative_prompt=None, seed=FIRST_SEED + i)
        for i in range(count)
    ]


def load_cases(path: Path) -> list[Case]:
    """Cases from a run file written here, a golden set from the validator, or a bare list of cases."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "cases" in data:  # a run file
            return [Case.from_json(case) for case in data["cases"]]
        if "entries" in data:  # a validator golden set
            return [Case.from_json(entry["case"]) for entry in data["entries"]]
        raise VerifiedCheckError(f"{path}: not a run file or a golden set (no 'cases' or 'entries')")
    return [Case.from_json(case) for case in data]


def run_case(backend: Backend, profile: ModelProfile, case: Case, workdir: Path, *, repeat: int = 0) -> dict[str, Any]:
    """Renders one case in verified mode and returns its committed leaves. Raises nothing a caller can't record."""
    from .bench import neutral_png  # a shared image for modes that need one

    job_id = f"verified-check-{case.name}-{repeat}"
    task = build_task(profile, case.params, workdir, seed=case.seed, prompt=case.prompt,
                      negative_prompt=case.negative_prompt, job_id=job_id)
    for item in task.inputs:
        if item.ref.role in (InputRole.REFERENCE_IMAGE, InputRole.FIRST_FRAME, InputRole.LAST_FRAME, InputRole.KEYFRAME):
            item.data = neutral_png(task.width, task.height)
            item.ref = item.ref.model_copy(update={"size": len(item.data), "sha256": hashlib.sha256(item.data).hexdigest()})
    record: dict[str, Any] = {"name": case.name, "repeat": repeat, "seed": case.seed, "outcome": "ok", "detail": None}
    began = time.monotonic()
    try:
        result = backend.generate(task, lambda _value, _stage: None)
    except Exception as exc:  # a failure is a result: record it and let the other cases run
        return record | {"outcome": "error", "detail": f"{type(exc).__name__}: {exc}"[:500],
                         "wall_s": round(time.monotonic() - began, 2)}
    if result.openings is None:
        return record | {"outcome": "not_verified", "wall_s": round(time.monotonic() - began, 2),
                         "detail": f"{profile.id} produced no openings: verified mode is off for this hardware class"}
    retained = result.openings.store.record(job_id)
    try:
        if retained is None:
            return record | {"outcome": "not_verified", "wall_s": round(time.monotonic() - began, 2),
                             "detail": "the retention store kept nothing for this job"}
        transcript = retained.transcript
        return record | {
            "wall_s": round(time.monotonic() - began, 2),
            "runtime": transcript.runtime,
            "model_digest": transcript.model_digest,
            "conditioning_digest": transcript.conditioning_digest,
            "noise": transcript.noise,
            "determinism": transcript.determinism,
            "root": retained.commitment.root,
            "latent_shape": retained.commitment.latent_shape,
            "dtype": retained.commitment.dtype,
            "leaves": [{"index": leaf.index, "stage": leaf.stage, "kind": leaf.kind, "sigma": leaf.sigma, "latent": leaf.latent}
                       for leaf in retained.leaves],
        }
    finally:
        result.openings.discard()


def run(backend: Backend, profile: ModelProfile, cases: list[Case], workdir: Path, *, repeats: int = 1) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        for repeat in range(repeats):
            record = run_case(backend, profile, case, workdir, repeat=repeat)
            log.info("%s repeat %d: %s (%s leaves)", case.name, repeat, record["outcome"], len(record.get("leaves") or []))
            results.append(record)
    return results


# ------------------------------------------------------------------ comparison


def leaves_of(doc: dict[str, Any], name: str, repeat: int = 0) -> dict[str, Any] | None:
    return next((r for r in doc["runs"] if r["name"] == name and r["repeat"] == repeat), None)


def compare_runs(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """(identical, differences, notes). Differences are divergences; notes are context worth printing either way."""
    differences: list[str] = []
    notes: list[str] = []
    for field in ("profile_id", "hardware_class"):
        if a.get(field) != b.get(field):
            raise VerifiedCheckError(f"the runs are not comparable: {field} is {a.get(field)!r} and {b.get(field)!r}")
    names_a = [c["name"] for c in a["cases"]]
    names_b = [c["name"] for c in b["cases"]]
    if names_a != names_b:
        raise VerifiedCheckError("the runs used different cases; give the second run --cases <the first run's file>")
    for case_a, case_b in zip(a["cases"], b["cases"]):
        if case_a != case_b:
            raise VerifiedCheckError(f"case {case_a['name']} differs between the runs (prompt, seed or params)")
    machine_a, machine_b = a.get("machine") or {}, b.get("machine") or {}
    if machine_a.get("gpu_model") != machine_b.get("gpu_model"):
        notes.append(f"different GPUs: {machine_a.get('gpu_model')} and {machine_b.get('gpu_model')} "
                     "(a cross-machine comparison, Phase 0 step 2)")
    salted = 0  # identical leaves under different Merkle roots: every job salts its own tree, so this is the norm
    for name in names_a:
        first, second = leaves_of(a, name), leaves_of(b, name)
        if first is None or second is None:
            differences.append(f"{name}: missing from one run")
            continue
        if first["outcome"] != "ok" or second["outcome"] != "ok":
            differences.append(f"{name}: {first['outcome']} and {second['outcome']} ({first.get('detail') or second.get('detail')})")
            continue
        if first.get("model_digest") != second.get("model_digest"):
            raise VerifiedCheckError(f"{name}: different weights ({first.get('model_digest')} and {second.get('model_digest')})")
        if first.get("conditioning_digest") != second.get("conditioning_digest"):
            differences.append(f"{name}: the conditioning differs (the text encoder is not deterministic here)")
            continue
        leaves_a, leaves_b = first["leaves"], second["leaves"]
        if len(leaves_a) != len(leaves_b):
            differences.append(f"{name}: {len(leaves_a)} leaves and {len(leaves_b)} (the schedule changed)")
            continue
        diverged = next((i for i, (x, y) in enumerate(zip(leaves_a, leaves_b)) if x["latent"] != y["latent"]), None)
        if diverged is None:
            if first["root"] != second["root"]:
                salted += 1
            continue
        leaf = leaves_a[diverged]
        where = "the seed's initial noise" if leaf["kind"] == "init" else f"stage {leaf['stage']}, sigma {leaf['sigma']}"
        differences.append(f"{name}: diverges at leaf {diverged} of {len(leaves_a)} ({where})")
    if salted:
        notes.append(f"{salted} of {len(names_a)} cases: identical leaves under different roots, as expected (each job salts its own tree)")
    return not differences, differences, notes


# ------------------------------------------------------------------ CLI


def machine_record(kind: str, profile: ModelProfile) -> dict[str, Any]:
    from .bench_probes import probe_machine, simulated_machine

    if kind != "real":
        return simulated_machine("simulated GPU", profile.gpus_per_worker)
    try:
        return probe_machine()[1]
    except Exception as exc:  # nvidia-smi missing or refusing: the run still happened, so record why not
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kuno-verified-check", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="render the golden cases in verified mode and write their committed leaves")
    run_cmd.add_argument("--backend", choices=("real", "mock"), default="real")
    run_cmd.add_argument("--profile", help="profile id (default: the profile of --cases)")
    run_cmd.add_argument("--hardware-class", help="verified hardware class (default: what --cases used)")
    run_cmd.add_argument("--cases", type=Path, help="a previous run file, a golden set, or a list of cases")
    run_cmd.add_argument("--case-count", type=int, default=DEFAULT_CASES, help=f"how many default cases (default {DEFAULT_CASES})")
    run_cmd.add_argument("--repeats", type=int, default=1, help="renders per case in this process (default 1; two processes are the real check)")
    run_cmd.add_argument("--duration", type=float, dest="duration_s", help="override the default case duration in seconds")
    run_cmd.add_argument("--resolution")
    run_cmd.add_argument("--aspect", dest="aspect_ratio")
    run_cmd.add_argument("--fps", type=int)
    run_cmd.add_argument("--models-dir", type=Path, help="LTX weights (KUNO_LTX_MODELS_DIR)")
    run_cmd.add_argument("--model-digest", help="expected weights digest (KUNO_MODEL_DIGEST)")
    run_cmd.add_argument("--offload", help="LTX offload mode (KUNO_LTX_OFFLOAD)")
    run_cmd.add_argument("--weights-verify", choices=("full", "size"), help="KUNO_WEIGHTS_VERIFY (full hashes the weights inside the cold load)")
    run_cmd.add_argument("--h3-turbo-lora", help="path to the h3-turbo LoRA")
    run_cmd.add_argument("--allow-unpinned", action="store_true", help="accept weights without a pinned digest")
    run_cmd.add_argument("--workdir", type=Path, help="scratch directory (default: a temporary one)")
    run_cmd.add_argument("--out", type=Path, required=True)

    compare = sub.add_parser("compare", help="compare two or more run files leaf by leaf")
    compare.add_argument("runs", type=Path, nargs="+")
    compare.add_argument("--json", dest="as_json", action="store_true", help="machine-readable verdict on stdout")
    return parser


def _run(args: argparse.Namespace) -> int:
    profiles = load_profiles()
    cases = load_cases(args.cases) if args.cases else None
    profile_id = args.profile or (cases[0].profile_id if cases else None)
    if not profile_id:
        raise VerifiedCheckError("give --profile or --cases")
    if profile_id not in profiles:
        raise VerifiedCheckError(f"unknown profile {profile_id}")
    profile = profiles[profile_id]
    hardware_class = args.hardware_class
    if hardware_class is None and args.cases:  # a run file says which class it ran on; a bare case list does not
        previous = json.loads(args.cases.read_text())
        hardware_class = previous.get("hardware_class") if isinstance(previous, dict) else None
    if not hardware_class:
        raise VerifiedCheckError("give --hardware-class (the class this machine declares); verified mode is off without one")
    if profile.verified is None or profile.verified.hardware_class(hardware_class) is None:
        raise VerifiedCheckError(f"{profile_id} has no verified hardware class {hardware_class}")
    args.hardware_class = hardware_class  # the class the real backend is built with, whether it was given or read back
    if cases is None:
        cases = default_cases(profile, args.case_count, duration_s=args.duration_s, resolution=args.resolution,
                              aspect_ratio=args.aspect_ratio, fps=args.fps)
    for case in cases:
        if case.profile_id != profile_id:
            raise VerifiedCheckError(f"case {case.name} is for {case.profile_id}, not {profile_id}")
        try:
            check_prompt(case.prompt)
        except ContentPolicyViolation as exc:
            raise VerifiedCheckError(f"case {case.name}'s prompt fails the content policy: {exc}") from None

    with tempfile.TemporaryDirectory(prefix="kuno-verified-check-") as tmp:
        workdir = args.workdir or Path(tmp)
        workdir.mkdir(parents=True, exist_ok=True)
        if args.backend == "mock":
            from .bench_probes import Clock, SimulatedMemory, StepTimer, simulated_backends

            backends = simulated_backends(workdir, Clock(), StepTimer(), SimulatedMemory(), hardware_class=hardware_class, speed=1000.0)
        else:
            from .bench import real_backends

            backends = real_backends(args, [profile_id], workdir)
        backend = backends.get(profile.family) or backends.get("*")
        if backend is None:
            raise VerifiedCheckError(f"no backend for {profile.family}")
        if not backend.verified_enabled(profile):
            raise VerifiedCheckError(f"the backend runs {profile_id} with verified mode off on {hardware_class}")
        backend.warm(profile)
        runs = run(backend, profile, cases, workdir, repeats=args.repeats)

    doc = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "created_at": time.time(),
        "profile_id": profile_id, "family": profile.family, "hardware_class": hardware_class,
        "runtime": profile.verified.runtime, "backend": args.backend,
        "comparison": profile.verified.hardware_class(hardware_class).comparison,
        "machine": machine_record(args.backend, profile),
        "cases": [case.as_json() for case in cases],
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    failed = [r for r in runs if r["outcome"] != "ok"]
    print(f"{len(runs) - len(failed)} of {len(runs)} renders committed a trajectory -> {args.out}")
    for record in failed:
        print(f"  {record['name']} repeat {record['repeat']}: {record['outcome']}: {record['detail']}")
    if args.repeats > 1 and not failed:
        # Repeats in one process are the weaker check (Phase 0 wants two processes), but they cost nothing to compare.
        first = doc | {"runs": [r for r in runs if r["repeat"] == 0]}
        later = doc | {"runs": [dict(r, repeat=0) for r in runs if r["repeat"] == 1]}
        identical, differences, _ = compare_runs(first, later)
        print("  repeats within this process: " + ("identical" if identical else "; ".join(differences)))
    return 0 if not failed else 1


def _compare(args: argparse.Namespace) -> int:
    docs = []
    for path in args.runs:
        doc = json.loads(path.read_text())
        if doc.get("schema") != SCHEMA:
            raise VerifiedCheckError(f"{path} is not a {SCHEMA} file")
        docs.append((path, doc))
    verdicts = []
    identical = True
    for (path_a, first), (path_b, second) in zip(docs, docs[1:]):
        same, differences, notes = compare_runs(first, second)
        identical = identical and same
        verdicts.append({"a": str(path_a), "b": str(path_b), "identical": same, "differences": differences, "notes": notes})
    leaves = sum(len(r.get("leaves") or []) for r in docs[0][1]["runs"])
    result = {"identical": identical, "runs": [str(p) for p, _ in docs], "cases": len(docs[0][1]["cases"]),
              "leaves_per_run": leaves, "hardware_class": docs[0][1]["hardware_class"], "comparisons": verdicts}
    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        head = f"{docs[0][1]['profile_id']} on {docs[0][1]['hardware_class']}: {len(docs[0][1]['cases'])} cases, {leaves} leaves per run"
        print(head)
        for verdict in verdicts:
            print(f"  {Path(verdict['a']).name} vs {Path(verdict['b']).name}: " + ("identical" if verdict["identical"] else "DIVERGED"))
            for line in verdict["differences"]:
                print(f"    {line}")
            for line in verdict["notes"]:
                print(f"    note: {line}")
        print("deterministic" if identical else "NOT deterministic: this class cannot earn in verified mode yet")
    return 0 if identical else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        return _run(args) if args.command == "run" else _compare(args)
    except VerifiedCheckError as exc:
        print(f"kuno-verified-check: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
