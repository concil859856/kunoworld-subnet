"""kuno-safety-check: load the content safety classifiers exactly as the worker does, and run each once.

A self-test for a worker image or a machine, not an evaluation. It builds the safety gate from the
environment (the KUNO_SAFETY_* settings, see safety.py) and fails when a prompt or frame classifier is
missing or did not load, as a production (TDX) worker refuses to start. Then it classifies one fixed
benign prompt and one synthetic frame on CPU, and prints every model's scores and timings as JSON.
It sees no customer data.

    docker run --rm --entrypoint kuno-safety-check kuno-worker:ltx

Exits 0 when every classifier loaded, answered, and let the benign inputs through; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

from .safety import SafetyGate, SafetyViolation

# Fixed benign inputs. Passing shows the models load and answer, and that they do not block everything.
PROMPT = "A red kite flies over a green hill on a sunny afternoon."


def synthetic_frame(size: int = 256) -> Any:
    """An HxWx3 uint8 RGB gradient: no people and nothing any policy category describes."""
    import numpy as np  # noqa: PLC0415

    ramp = np.linspace(0, 255, size).astype(np.uint8)
    frame = np.empty((size, size, 3), dtype=np.uint8)
    frame[..., 0] = ramp[None, :]
    frame[..., 1] = ramp[:, None]
    frame[..., 2] = 128
    return frame


def _name(model: Any) -> str:
    return str(getattr(model, "name", type(model).__name__))


def run(gate: SafetyGate) -> tuple[dict[str, Any], list[str]]:
    """Exercises a gate built from the environment: (report, reasons it failed, empty when it passed)."""
    report: dict[str, Any] = {"gate": gate.status()}
    problems = list(gate.startup_errors(required=True))
    if problems:
        return report, problems

    started = time.perf_counter()
    try:
        scores = {category: float(score) for category, score in gate.classifier.classify(PROMPT).items()}
    except Exception as exc:  # the input is a constant, so the message is safe to print
        return report, [f"prompt classifier {_name(gate.classifier)} failed ({type(exc).__name__}: {exc})"]
    report["prompt"] = {"classifier": _name(gate.classifier), "scores": scores, "seconds": round(time.perf_counter() - started, 3)}
    try:
        gate._judge(scores)
    except SafetyViolation as violation:
        problems.append(f"the benign prompt was blocked as {getattr(violation, 'category', 'unknown')}")

    frame = synthetic_frame()
    rows: list[Any] = []
    report["frame"] = []
    for model in gate._frame_models():
        started = time.perf_counter()
        try:
            model_rows = gate._frame_rows(model, [frame])
        except Exception as exc:
            problems.append(f"frame classifier {_name(model)} failed ({type(exc).__name__}: {exc})")
            continue
        report["frame"].append(
            {"classifier": _name(model), "scores": [dict(row) for row in model_rows], "seconds": round(time.perf_counter() - started, 3)}
        )
        rows.extend(model_rows)
    try:
        category = gate.frame_policy.decide(rows)
    except ValueError as exc:
        problems.append(f"a frame classifier returned an invalid score ({exc})")
    else:
        if category is not None:
            problems.append(f"the synthetic frame was blocked as {category}")
    return report, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kuno-safety-check", description="Load the safety classifiers from KUNO_SAFETY_* and run each once on synthetic inputs.")
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.perf_counter()
    try:
        gate = SafetyGate.from_env()
    except ValueError as exc:  # malformed settings, e.g. a threshold above its ceiling
        print(json.dumps({"ok": False, "problems": [f"invalid safety settings: {exc}"]}, indent=2))
        return 1
    load_seconds = round(time.perf_counter() - started, 3)
    report, problems = run(gate)
    report.update(ok=not problems, problems=problems, load_seconds=load_seconds)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
