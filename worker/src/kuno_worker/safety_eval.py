"""kuno-safety-eval: how often the safety gate blocks what it should, and how often it blocks what it shouldn't.

    kuno-safety-eval prompts --data prompts.jsonl --out prompts-report.json
    kuno-safety-eval prompts --data aegis-test.json --format aegis --out aegis-report.json
    kuno-safety-eval videos --data videos.jsonl --out videos-report.json

The gate is built from the environment exactly as the worker builds it (safety.SafetyGate.from_env), so a run inside a
worker image measures that image's classifiers, weights and thresholds. Each example is decided the way a job is:
- prompts go through the shared content policy, then the prompt classifier and its thresholds;
- videos go through frame sampling, the frame classifiers and FramePolicy.

The report counts true and false positives and negatives for each stage and for the gate as a whole, recall per
labelled category, the ids of every miss and every false block, the highest score per category, and latency. It
never contains prompt text or frames: an id and a label are all that leave the loop.

**Data.**
- **Text.** Prompt sets are text, and public safety benchmarks exist:
  - ToxicChat (`lmsys/toxic-chat`, CC-BY-NC-4.0; `--format toxicchat` reads its CSV).
  - Aegis 2.0 (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`, CC-BY-4.0; `--format aegis` reads `test.json`).
  - Any JSONL of `{"id", "text", "unsafe": bool, "category"?}` (`--format kuno`).
  - Their "unsafe" is broader than KunoWorld's policy, which bans all sexual content and blocks other harms only at
    the classifier's thresholds. So read recall per category, and the false-positive rate on benign prompts.
- **Video.** A video set is a JSONL of `{"id", "path", "violating": bool, "category"?}`. Measure false positives on
  benign video freely (e.g. the network's own renders). Measure detection only on a vetted evaluation set handled under
  the subnet owner's legal process. Never download, generate or keep sexual or abusive imagery to exercise this tool.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kuno_protocol.content_policy import ContentPolicyViolation

from .safety import SafetyGate, SafetyUnavailable
from .safety_frames import POLICY_CATEGORIES, RequestSignals, sample_frames

log = logging.getLogger("kuno.safety_eval")

SCHEMA = "kuno-safety-eval"
SCHEMA_VERSION = 1


@dataclass
class Example:
    id: str
    unsafe: bool
    category: str | None
    text: str | None = None
    path: Path | None = None


# ---------------------------------------------------------------- loading


def load_prompts(path: Path, fmt: str, limit: int | None = None) -> list[Example]:
    rows: Iterator[Example]
    if fmt == "kuno":
        rows = (Example(str(r["id"]), bool(r["unsafe"]), r.get("category"), text=str(r["text"])) for r in _jsonl(path))
    elif fmt == "toxicchat":
        rows = (
            Example(str(r["conv_id"]), str(r["toxicity"]).strip() == "1", "toxic" if str(r["toxicity"]).strip() == "1" else None,
                    text=r["user_input"])
            for r in csv.DictReader(path.open(newline="", encoding="utf-8"))
        )
    elif fmt == "aegis":
        records = json.loads(path.read_text()) if path.suffix == ".json" else list(_jsonl(path))
        rows = (
            Example(str(r["id"]), str(r["prompt_label"]).lower() == "unsafe",
                    (str(r.get("violated_categories") or "").split(",")[0].strip() or None), text=r["prompt"])
            for r in records
            if r.get("prompt") and r.get("prompt") != "REDACTED"  # redacted rows ship without their text
        )
    else:
        raise ValueError(f"unknown prompt format {fmt!r}")
    out = []
    for example in rows:
        out.append(example)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_videos(path: Path) -> list[Example]:
    base = path.parent
    out = []
    for row in _jsonl(path):
        video = Path(row["path"])
        out.append(Example(str(row["id"]), bool(row["violating"]), row.get("category"), path=video if video.is_absolute() else base / video))
    return out


def _jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------- counting


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, unsafe: bool, blocked: bool) -> None:
        if unsafe and blocked:
            self.tp += 1
        elif unsafe:
            self.fn += 1
        elif blocked:
            self.fp += 1
        else:
            self.tn += 1

    def summary(self) -> dict[str, Any]:
        def ratio(a: int, b: int) -> float | None:
            return round(a / b, 4) if b else None

        precision, recall = ratio(self.tp, self.tp + self.fp), ratio(self.tp, self.tp + self.fn)
        f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else None
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn, "precision": precision, "recall": recall,
                "false_positive_rate": ratio(self.fp, self.fp + self.tn), "f1": f1}


@dataclass
class Tally:
    stages: dict[str, Confusion] = field(default_factory=dict)
    by_category: dict[str, list[int]] = field(default_factory=dict)  # category -> [blocked, total] over unsafe examples
    misses: list[str] = field(default_factory=list)
    false_blocks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    max_scores: dict[str, float] = field(default_factory=dict)
    # id -> the highest score per category: numbers only, so near misses can be found without the content.
    scores: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, example: Example, decisions: dict[str, str | None], seconds: float, scores: dict[str, float]) -> None:
        """`decisions`: stage -> the category that stage blocked for, or None. The gate blocks if any stage did."""
        gate = next((c for c in decisions.values() if c), None)
        for stage, category in [*decisions.items(), ("gate", gate)]:
            self.stages.setdefault(stage, Confusion()).add(example.unsafe, category is not None)
        if example.unsafe:
            counts = self.by_category.setdefault(example.category or "unlabelled", [0, 0])
            counts[0] += gate is not None
            counts[1] += 1
            if gate is None:
                self.misses.append(example.id)
        elif gate is not None:
            self.false_blocks.append({"id": example.id, "blocked_as": gate})
        self.latencies.append(seconds)
        if scores:
            self.scores[example.id] = {category: round(float(score), 4) for category, score in sorted(scores.items())}
        for category, score in scores.items():
            self.max_scores[category] = max(self.max_scores.get(category, 0.0), round(float(score), 4))

    def report(self) -> dict[str, Any]:
        latencies = sorted(self.latencies)
        return {
            "examples": len(self.latencies) + len(self.errors),
            "stages": {stage: confusion.summary() for stage, confusion in self.stages.items()},
            "recall_by_category": {c: {"blocked": b, "total": t, "recall": round(b / t, 4)} for c, (b, t) in sorted(self.by_category.items())},
            "misses": self.misses,
            "false_blocks": self.false_blocks,
            "errors": self.errors,
            "highest_scores": dict(sorted(self.max_scores.items())),
            "scores": self.scores,
            "seconds": {
                "median": round(statistics.median(latencies), 4) if latencies else None,
                "p95": round(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 4) if latencies else None,
            },
        }


# ---------------------------------------------------------------- deciding


def decide_prompt(gate: SafetyGate, text: str) -> tuple[dict[str, str | None], dict[str, float]]:
    decisions: dict[str, str | None] = {"content_policy": None}
    try:
        gate.blocklist.check(text)
    except ContentPolicyViolation as violation:
        decisions["content_policy"] = getattr(violation, "category", None) or "content_policy"
    scores: dict[str, float] = {}
    if gate.classifier is not None:
        scores = {k: float(v) for k, v in gate._score(gate.classifier.classify, text).items()}
        try:
            gate._judge(scores)
            decisions["classifier"] = None
        except ContentPolicyViolation as violation:
            decisions["classifier"] = getattr(violation, "category", None) or "classifier"
    return decisions, scores


def decide_video(gate: SafetyGate, video: bytes, signals: RequestSignals | None = None) -> tuple[dict[str, str | None], dict[str, float]]:
    models = gate._frame_models()
    if not models:
        raise SafetyUnavailable()
    size = max(int(getattr(model, "input_size", 224)) for model in models)
    frames = sample_frames(video, gate.frames_to_sample, size)
    if not frames:
        raise SafetyUnavailable()
    rows = []
    for model in models:
        rows.extend(gate._score(lambda f, m=model: gate._frame_rows(m, f), frames))
    scores: dict[str, float] = {}
    for row in rows:
        for category, value in row.items():
            if category in POLICY_CATEGORIES:
                scores[category] = max(scores.get(category, 0.0), float(value))
    return {"frames": gate.frame_policy.decide(rows, signals)}, scores


def evaluate(examples: Iterable[Example], decide) -> Tally:
    tally = Tally()
    for example in examples:
        started = time.perf_counter()
        try:
            decisions, scores = decide(example)
        except (SafetyUnavailable, OSError, ValueError, RuntimeError) as exc:
            # The type only: a message could quote the input.
            tally.errors.append({"id": example.id, "error": type(exc).__name__})
            continue
        tally.record(example, decisions, time.perf_counter() - started, scores)
    return tally


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kuno-safety-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prompts = sub.add_parser("prompts", help="the content policy and prompt classifier over labelled prompts")
    prompts.add_argument("--data", type=Path, required=True)
    prompts.add_argument("--format", choices=("kuno", "toxicchat", "aegis"), default="kuno")
    prompts.add_argument("--limit", type=int)
    prompts.add_argument("--out", type=Path, required=True)
    videos = sub.add_parser("videos", help="frame sampling, frame classifiers and FramePolicy over labelled videos")
    videos.add_argument("--data", type=Path, required=True)
    videos.add_argument("--out", type=Path, required=True)
    return parser


def summary_lines(report: dict[str, Any]) -> list[str]:
    lines = [f"{report['kind']}: {report['examples']} examples, {len(report['errors'])} errors, gate {report['gate']}"]
    for stage, s in report["stages"].items():
        lines.append(
            f"  {stage:15s} tp={s['tp']:<5d} fp={s['fp']:<5d} tn={s['tn']:<5d} fn={s['fn']:<5d} "
            f"precision={s['precision']} recall={s['recall']} false_positive_rate={s['false_positive_rate']}"
        )
    for category, c in report["recall_by_category"].items():
        lines.append(f"  recall {category}: {c['blocked']}/{c['total']} ({c['recall']})")
    if report["false_blocks"]:
        blocks = ", ".join("{} ({})".format(b["id"], b["blocked_as"]) for b in report["false_blocks"][:20])
        lines.append(f"  false blocks: {blocks}")
    lines.append(f"  highest scores: {report['highest_scores']}; seconds median {report['seconds']['median']}, p95 {report['seconds']['p95']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    gate = SafetyGate.from_env()
    if args.command == "prompts":
        examples = load_prompts(args.data, args.format, args.limit)
        tally = evaluate(examples, lambda e: decide_prompt(gate, e.text or ""))
    else:
        examples = load_videos(args.data)
        tally = evaluate(examples, lambda e: decide_video(gate, e.path.read_bytes()))
    report = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "kind": args.command, "data": str(args.data),
              "gate": gate.status(), "created_at": time.time(), **tally.report()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("\n".join(summary_lines(report)))
    return 0 if not tally.errors else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
