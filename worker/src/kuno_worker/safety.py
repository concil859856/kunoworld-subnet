"""Content safeguards that run inside the enclave.

Nobody outside the VM can see prompts or outputs, so the safety filter has to
live here, baked into every attested image. The MiniMax H3 license also requires
hosted services to maintain safeguards against its Acceptable Use Policy.

All sexual content (NSFW) is banned in both Private and Standard mode. The gate is a pipeline:
  1. the shared content policy, `kuno_protocol.content_policy.check_prompt`: a deterministic
     blocklist over normalized text, hardened against the usual obfuscations (homoglyphs,
     leetspeak, zero-width characters, spaced or dotted letters, repeated letters), with
     allow-contexts for ordinary phrases and co-occurrence rules for minors and deepfakes. The
     gateway runs the same function, so both enforce one list;
  2. a pluggable prompt classifier (recommended: Qwen3Guard-Gen-0.6B, Apache-2.0,
     loaded from a local path, CPU only; see `Qwen3GuardClassifier`). Its sexual categories
     always block: thresholds for them can be lowered, never raised past BANNED_CEILINGS;
  3. frame classifiers over frames sampled from the finished video, before it is sealed
     (sexual and sexualised content, and apparent minors; see `safety_frames`). There is no
     setting that allows sexual content.

Contract with worker.py: `check_request(prompt, negative_prompt)` returns None or raises
`SafetyViolation`, which the worker reports as `safety_blocked`; `check_output(video, signals, shot_frames)`
does the same for the rendered MP4. When a configured classifier cannot give an answer, the
gate raises `SafetyUnavailable` instead — not a SafetyViolation — so the job fails closed as
the miner's `internal_error` rather than being blamed on the customer.

Nothing here may put prompt text or frames into logs, exception messages or tracebacks.

Configuration (environment, read once):
  KUNO_SAFETY_CLASSIFIER           qwen3guard | sequence | none      (unset: blocklist only, logged as an error)
  KUNO_SAFETY_MODEL_PATH           local directory with the classifier weights (never downloaded)
  KUNO_SAFETY_REQUIRE_CLASSIFIER   1 to refuse to start without a working prompt classifier and frame classifier
  KUNO_SAFETY_THRESHOLDS           JSON {category: score} overriding DEFAULT_THRESHOLDS (sexual ones may only be lowered)
  KUNO_SAFETY_LABEL_MAP            JSON {model label: category} for the `sequence` adapter
  KUNO_SAFETY_FRAME_MODEL_PATH     sexual-content image classifier directory (unset: outputs unchecked, logged as an error)
  KUNO_SAFETY_MINOR_MODEL_PATH     CLIP directory for apparent-minor presence (unset: minors assumed in every frame)
  KUNO_SAFETY_FRAMES               frames sampled per video, first and last included (default 10; a storyboard: at
                                   least 10, plus 3 inside every shot)
  KUNO_SAFETY_FRAME_THRESHOLDS     JSON lowering FramePolicy thresholds: sexual, suggestive, minor, minor_sexual, minor_suggestive
  KUNO_SAFETY_FRAME_LABEL_MAP, KUNO_SAFETY_FRAME_DTYPE, KUNO_SAFETY_THREADS   see safety_frames.load_frame_models
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from kuno_protocol import content_policy
from kuno_protocol.content_policy import ContentPolicyViolation

from .safety_frames import POLICY_CATEGORIES, FramePolicy, RequestSignals, load_frame_models, sample_frames

log = logging.getLogger("kuno.worker.safety")

STORYBOARD_MIN_FRAMES = 10

# The request or output breaks the acceptable use policy. The same class the shared content policy
# raises, so a prompt blocked by the list and a video blocked by the frame check are handled alike.
# `category` is for tests and aggregate counters only; the message never varies, so nothing about
# the request can leak through it.
SafetyViolation = ContentPolicyViolation


class SafetyUnavailable(RuntimeError):
    """A required classifier could not judge the request, so it must not run."""

    def __init__(self) -> None:
        super().__init__("content safety classifier unavailable")


class SafetyConfigError(ValueError):
    """The safety configuration forbids starting (e.g. a required classifier is missing)."""


# ---------------------------------------------------------------- stage 1: the shared content policy


class Blocklist:
    """Stage 1: `kuno_protocol.content_policy`, the same deterministic list the gateway enforces.

    The list lives in the protocol package so that the gateway and every worker image block exactly
    the same prompts; this wrapper only gives the gate something to hold (and tests something to swap).
    """

    def check(self, prompt: str, negative_prompt: str | None = None) -> None:
        content_policy.check_prompt(prompt, negative_prompt)

    def mentions_minor(self, prompt: str) -> bool:
        """True when the prompt (not the negative prompt, which lists what to avoid) names a minor."""
        return content_policy.mentions_minor(prompt)


# ---------------------------------------------------------------- stage 2: prompt classifier

# All sexual content is banned, so these categories block at no more than these scores whatever the
# configuration says: an override may lower them (stricter), never raise them.
BANNED_CEILINGS: dict[str, float] = {
    "sexual_minors": 0.3,
    "sexual": 0.5,
}

# A category blocks when its score reaches the threshold. Unlisted categories never block.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "sexual_minors": 0.3,
    "sexual": 0.5,
    "self_harm": 0.5,
    "violence": 1.0,
    "illegal": 1.0,
    "hate": 0.9,
    "unethical": 1.0,
    "jailbreak": 1.0,
}


class PromptClassifier(Protocol):
    name: str

    def classify(self, text: str) -> Mapping[str, float]:
        """Scores in [0, 1] per category. May raise; the gate then fails closed."""


class FrameClassifier(Protocol):
    name: str

    def classify_frames(self, frames: Sequence[Any]) -> Mapping[str, float]:
        """Scores in [0, 1] per category over HxWx3 uint8 RGB frames (the max over frames)."""


QWEN3GUARD_CATEGORIES = {
    "violent": "violence",
    "non-violent illegal acts": "illegal",
    "sexual content or sexual acts": "sexual",
    "pii": "pii",
    "suicide & self-harm": "self_harm",
    "unethical acts": "unethical",
    "politically sensitive topics": "political",
    "copyright violation": "copyright",
    "jailbreak": "jailbreak",
}
_QWEN3GUARD_LEVEL = {"unsafe": 1.0, "controversial": 0.5, "safe": 0.0}


def parse_qwen3guard(output: str) -> dict[str, float]:
    """Parses Qwen3Guard-Gen's "Safety: Unsafe / Categories: Violent" answer into scores.

    Unsafe scores 1.0 and Controversial 0.5, so the default thresholds block unsafe
    content in every listed category and also controversial sexual and self-harm content.
    An answer we cannot parse raises, which the gate treats as unavailable (fail closed).
    """
    level = re.search(r"safety:\s*(safe|unsafe|controversial)", output, re.IGNORECASE)
    if level is None:
        raise ValueError("unparseable guard output")
    score = _QWEN3GUARD_LEVEL[level.group(1).lower()]
    if score == 0.0:
        return {}
    categories = re.search(r"categories:\s*([^\n]*)", output, re.IGNORECASE)
    names = [c.strip().lower() for c in re.split(r",|;", categories.group(1))] if categories else []
    scores = {QWEN3GUARD_CATEGORIES[n]: score for n in names if n in QWEN3GUARD_CATEGORIES}
    # Unsafe with no recognized category still has to block something.
    return scores or {"unknown_unsafe": score}


class Qwen3GuardClassifier:
    """Qwen/Qwen3Guard-Gen-0.6B (Apache-2.0, https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B).

    About 0.75B parameters (1.5 GB in bf16), CPU-only, weights from a local directory
    baked into the worker images (image/CVM.md pins the revision). Needs the worker's `safety`
    extra. `kuno-safety-check` runs it against those weights on a benign prompt only; its
    accuracy, and its latency on the target CPU, are unmeasured.
    """

    name = "qwen3guard-gen"

    def __init__(self, model_path: Path, max_new_tokens: int = 32):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415 — optional extra

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype="auto")
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    def classify(self, text: str) -> Mapping[str, float]:
        import torch  # noqa: PLC0415

        chat = self._tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([chat], return_tensors="pt")
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        answer = self._tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return parse_qwen3guard(answer)


OPENAI_STYLE_LABELS = {
    "s": "sexual", "sexual": "sexual", "s3": "sexual_minors", "sexual/minors": "sexual_minors",
    "nsfw": "sexual", "porn": "sexual", "sexual_explicit": "sexual",
    "h": "hate", "hate": "hate", "h2": "hate", "hate/threatening": "hate",
    "v": "violence", "violence": "violence", "v2": "violence", "violence/graphic": "violence",
    "sh": "self_harm", "self-harm": "self_harm", "self_harm": "self_harm",
    "hr": "harassment", "harassment": "harassment",
}


class SequenceClassifier:
    """Any Hugging Face sequence-classification moderation model with named labels.

    For example oxyapi/albert-moderation-001 (Apache-2.0, 67M parameters), a lighter
    alternative to Qwen3Guard that has no published evaluation: measure it first.
    """

    name = "sequence"

    def __init__(self, model_path: Path, label_map: Mapping[str, str] | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415 — optional extra

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self._model.eval()
        self._labels = {int(i): str(label).lower() for i, label in self._model.config.id2label.items()}
        self._map = {k.lower(): v for k, v in (label_map or OPENAI_STYLE_LABELS).items()}
        self._multi_label = self._model.config.problem_type != "single_label_classification"

    def classify(self, text: str) -> Mapping[str, float]:
        import torch  # noqa: PLC0415

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            logits = self._model(**inputs).logits[0]
        probs = torch.sigmoid(logits) if self._multi_label else torch.softmax(logits, dim=-1)
        scores: dict[str, float] = {}
        for index, value in enumerate(probs.tolist()):
            category = self._map.get(self._labels.get(index, ""))
            if category:
                scores[category] = max(scores.get(category, 0.0), float(value))
        return scores


# ---------------------------------------------------------------- the gate


@dataclass
class SafetyGate:
    blocklist: Blocklist = field(default_factory=Blocklist)
    classifier: PromptClassifier | None = None
    frame_classifier: FrameClassifier | None = None
    thresholds: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    # Refuse all work when no prompt classifier is present. Production images set this.
    require_classifier: bool = False
    # Set when a classifier was configured but failed to load: the gate then fails closed.
    unavailable: bool = False
    # Per-frame scorers (`safety_frames.FrameScorer`); `frame_classifier` above is the older
    # whole-clip interface and is checked too.
    frame_classifiers: list[Any] = field(default_factory=list)
    frame_policy: FramePolicy = field(default_factory=FramePolicy)
    # Set when frame models were configured but failed to load.
    frame_unavailable: bool = False
    frames_to_sample: int = 10

    def check_request(self, prompt: str, negative_prompt: str | None = None) -> None:
        self.blocklist.check(prompt, negative_prompt)
        # Refuse before spending GPU time on a video the output stage could not judge.
        if self.unavailable or self.frame_unavailable:
            raise SafetyUnavailable()
        if self.require_classifier and (self.classifier is None or not self._frame_models()):
            raise SafetyUnavailable()
        if self.classifier is None:
            return
        # Negative prompts list what to avoid, so a classifier would misread them; stage 1 covers them.
        self._judge(self._score(self.classifier.classify, prompt))

    def request_signals(self, prompt: str, negative_prompt: str | None = None) -> RequestSignals:
        """Booleans the output stage uses to err toward blocking. Carries no prompt text."""
        return RequestSignals(mentions_minor=self.blocklist.mentions_minor(prompt))

    def check_output(self, video: bytes, signals: RequestSignals | None = None, shot_frames: Sequence[int] | None = None) -> None:
        """Classifies frames sampled from the finished MP4, before it is sealed or signed. A storyboard passes its frames
        per shot (`GenerationTask.shot_frames`), so every shot is sampled, however short."""
        models = self._frame_models()
        if self.frame_unavailable or (self.require_classifier and not models):
            raise SafetyUnavailable()
        if not models:
            return
        size = max(int(getattr(model, "input_size", 224)) for model in models)
        # A storyboard is checked on at least 10 frames whatever KUNO_SAFETY_FRAMES says (PROTOCOL.md, "Storyboards").
        count = max(self.frames_to_sample, STORYBOARD_MIN_FRAMES) if shot_frames else self.frames_to_sample
        try:
            frames = sample_frames(video, count, size, shot_frames)
        except Exception as exc:  # ffmpeg's stderr is not echoed; keep only the type
            log.error("sampling frames for the safety check failed with %s; failing closed", type(exc).__name__)
            raise SafetyUnavailable() from None
        if not frames:
            raise SafetyUnavailable()
        rows: list[Mapping[str, float]] = []
        for model in models:
            rows.extend(self._score(lambda f, m=model: self._frame_rows(m, f), frames))
        try:
            category = self.frame_policy.decide(rows, signals)
        except ValueError:
            log.error("a frame classifier returned an invalid score; failing closed")
            raise SafetyUnavailable() from None
        if category is not None:
            raise SafetyViolation(category)
        # Any other category a frame model reports goes through the ordinary thresholds.
        others: dict[str, float] = {}
        for row in rows:
            for key, value in row.items():
                if key not in POLICY_CATEGORIES:
                    others[key] = max(others.get(key, 0.0), value)
        self._judge(others)

    def _frame_models(self) -> list[Any]:
        return ([self.frame_classifier] if self.frame_classifier is not None else []) + list(self.frame_classifiers)

    @staticmethod
    def _frame_rows(model: Any, frames: Sequence[Any]) -> list[Mapping[str, float]]:
        if hasattr(model, "score_frames"):
            rows = list(model.score_frames(frames))
            if len(rows) != len(frames):  # a dropped frame would go unexamined
                raise ValueError("frame scorer returned the wrong number of rows")
            return rows
        return [model.classify_frames(frames)]

    @staticmethod
    def _score(fn, arg) -> Mapping[str, float]:
        try:
            return fn(arg)
        except Exception as exc:  # the message could echo the input; keep only the type
            log.error("safety classifier failed with %s; failing closed", type(exc).__name__)
            raise SafetyUnavailable() from None

    def _judge(self, scores: Mapping[str, float]) -> None:
        for category, score in scores.items():
            threshold = self.thresholds.get(category, 1.0 if category == "unknown_unsafe" else None)
            if category in BANNED_CEILINGS:  # no configuration or constructor argument can loosen the ban
                ceiling = BANNED_CEILINGS[category]
                threshold = ceiling if threshold is None else min(threshold, ceiling)
            if threshold is not None and score >= threshold:
                raise SafetyViolation(category)

    def status(self) -> dict[str, Any]:
        """What the gate enforces, for startup logs and preflight. Contains no request data."""
        return {
            "blocklist": True,
            "classifier": getattr(self.classifier, "name", None),
            "classifier_unavailable": self.unavailable,
            "require_classifier": self.require_classifier,
            "frame_classifier": getattr(self.frame_classifier, "name", None),
            "frame_classifiers": [getattr(m, "name", type(m).__name__) for m in self.frame_classifiers],
            "frame_classifier_unavailable": self.frame_unavailable,
            "frames_to_sample": self.frames_to_sample,
            "content_policy": "kuno_protocol.content_policy",
        }

    def startup_errors(self, required: bool = False) -> list[str]:
        """Reasons this gate must not serve at all: missing classifiers, when KUNO_SAFETY_REQUIRE_CLASSIFIER is set
        or the caller requires them (`required`, e.g. a production TDX worker)."""
        if not (self.require_classifier or required):
            return []
        why = "KUNO_SAFETY_REQUIRE_CLASSIFIER is set" if self.require_classifier else "classifiers are required"
        errors = []
        if self.classifier is None or self.unavailable:
            errors.append(f"{why} but no prompt classifier loaded (KUNO_SAFETY_CLASSIFIER)")
        if not self._frame_models() or self.frame_unavailable:
            errors.append(f"{why} but no frame classifier loaded (KUNO_SAFETY_FRAME_MODEL_PATH)")
        return errors

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SafetyGate:
        env = os.environ if env is None else env
        kind = env.get("KUNO_SAFETY_CLASSIFIER", "").strip().lower()
        gate = cls(require_classifier=env.get("KUNO_SAFETY_REQUIRE_CLASSIFIER", "") in ("1", "true", "yes"))
        if env.get("KUNO_SAFETY_THRESHOLDS"):
            overrides = json.loads(env["KUNO_SAFETY_THRESHOLDS"])
            if not isinstance(overrides, dict):
                raise ValueError("KUNO_SAFETY_THRESHOLDS must be a JSON object")
            for category, ceiling in BANNED_CEILINGS.items():
                value = overrides.get(category)
                if value is not None and (not isinstance(value, (int, float)) or not 0.0 <= value <= ceiling):
                    raise ValueError(f"KUNO_SAFETY_THRESHOLDS[{category!r}] must be a number in [0, {ceiling}]: sexual content is banned")
            gate.thresholds = {**DEFAULT_THRESHOLDS, **overrides}
        # An explicit KUNO_SAFETY_CLASSIFIER=none is a deliberate opt-out: stay quiet about missing frame models too.
        gate._load_frames(env, quiet=kind == "none")
        if kind in ("", "none"):
            if not kind:
                log.error("KUNO_SAFETY_CLASSIFIER is not set: only the blocklist protects this worker")
            if gate.require_classifier:
                log.error("KUNO_SAFETY_REQUIRE_CLASSIFIER is set but no classifier is configured: refusing all requests")
            return gate
        path = env.get("KUNO_SAFETY_MODEL_PATH")
        try:
            if not path or not Path(path).is_dir():
                raise FileNotFoundError("KUNO_SAFETY_MODEL_PATH must name a local model directory")
            if kind == "qwen3guard":
                gate.classifier = Qwen3GuardClassifier(Path(path))
            elif kind == "sequence":
                label_map = json.loads(env["KUNO_SAFETY_LABEL_MAP"]) if env.get("KUNO_SAFETY_LABEL_MAP") else None
                gate.classifier = SequenceClassifier(Path(path), label_map)
            else:
                raise ValueError(f"unknown KUNO_SAFETY_CLASSIFIER {kind!r}")
        except Exception as exc:
            log.error("safety classifier %r failed to load (%s: %s); refusing all requests", kind, type(exc).__name__, exc)
            gate.unavailable = True
        return gate

    def _load_frames(self, env: Mapping[str, str], quiet: bool = False) -> None:
        """Frame policy and models. Malformed settings raise ValueError; models that fail to load fail closed."""
        self.frame_policy = FramePolicy.from_env(env)
        self.frames_to_sample = int(env.get("KUNO_SAFETY_FRAMES", "10"))
        if self.frames_to_sample < 2:
            raise ValueError("KUNO_SAFETY_FRAMES must be at least 2 (the first and last frame)")
        try:
            models, warnings = load_frame_models(env)
        except Exception as exc:  # loading sees no customer content, so the message is safe to print
            log.error("frame safety classifier failed to load (%s: %s); refusing all requests", type(exc).__name__, exc)
            self.frame_unavailable = True
            return
        if not models and not quiet:
            log.error("KUNO_SAFETY_FRAME_MODEL_PATH is not set: finished videos are not checked before delivery")
        for warning in warnings:
            log.error("%s", warning)
        self.frame_classifiers = list(models)


# ---------------------------------------------------------------- module interface used by worker.py

_gate: SafetyGate | None = None
_lock = threading.Lock()


def default_gate() -> SafetyGate:
    """The process-wide gate, built from the environment on first use (call early to load models at startup).

    Raises SafetyConfigError (a ValueError, so kuno-worker exits) when the configuration forbids serving.
    """
    global _gate
    with _lock:
        if _gate is None:
            gate = SafetyGate.from_env()
            errors = gate.startup_errors()
            if errors:
                raise SafetyConfigError("; ".join(errors))
            _gate = gate
        return _gate


def configure(gate: SafetyGate | None) -> None:
    """Installs a gate (tests, or a worker that builds its own); None rebuilds from the environment."""
    global _gate
    with _lock:
        _gate = gate


def check_request(prompt: str, negative_prompt: str | None = None) -> None:
    default_gate().check_request(prompt, negative_prompt)


def request_signals(prompt: str, negative_prompt: str | None = None) -> RequestSignals:
    return default_gate().request_signals(prompt, negative_prompt)


def check_output(video: bytes, signals: RequestSignals | None = None, shot_frames: Sequence[int] | None = None) -> None:
    default_gate().check_output(video, signals, shot_frames)
