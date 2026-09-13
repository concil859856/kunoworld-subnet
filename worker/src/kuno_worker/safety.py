"""Content safeguards that run inside the enclave.

Nobody outside the VM can see prompts or outputs, so the safety filter has to
live here, baked into every attested image. The MiniMax H3 license also requires
hosted services to maintain safeguards against its Acceptable Use Policy.

The gate is a pipeline:
  1. a deterministic blocklist over normalized text, hardened against the usual
     obfuscations (homoglyphs, leetspeak, zero-width characters, spaced or dotted
     letters, repeated letters), with co-occurrence rules for sexual content
     involving minors;
  2. a pluggable prompt classifier (recommended: Qwen3Guard-Gen-0.6B, Apache-2.0,
     loaded from a local path, CPU only; see `Qwen3GuardClassifier`);
  3. an optional classifier over frames sampled from the finished video.

Contract with worker.py (unchanged): `check_request(prompt, negative_prompt)` returns
None or raises `SafetyViolation`, which the worker reports as `safety_blocked`. When a
configured classifier cannot give an answer, the gate raises `SafetyUnavailable`
instead — not a SafetyViolation — so the job fails closed as the miner's
`internal_error` rather than being blamed on the customer.

Nothing here may put prompt text into logs, exception messages or tracebacks.

Configuration (environment, read once):
  KUNO_SAFETY_CLASSIFIER           qwen3guard | sequence | none      (unset: blocklist only, logged as an error)
  KUNO_SAFETY_MODEL_PATH           local directory with the classifier weights (never downloaded)
  KUNO_SAFETY_REQUIRE_CLASSIFIER   1 to refuse every request when no classifier is configured
  KUNO_SAFETY_THRESHOLDS           JSON {category: score} overriding DEFAULT_THRESHOLDS
  KUNO_SAFETY_LABEL_MAP            JSON {model label: category} for the `sequence` adapter
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("kuno.worker.safety")


class SafetyViolation(Exception):
    """The request or output breaks the acceptable use policy.

    `category` is for tests and aggregate counters only; the message never varies,
    so nothing about the request can leak through it.
    """

    def __init__(self, category: str = "policy"):
        super().__init__("request violates the acceptable use policy")
        self.category = category


class SafetyUnavailable(RuntimeError):
    """A required classifier could not judge the request, so it must not run."""

    def __init__(self) -> None:
        super().__init__("content safety classifier unavailable")


# ---------------------------------------------------------------- stage 1: normalized blocklist

_CONFUSABLES = str.maketrans(
    {
        # Cyrillic and Greek letters that render like Latin ones.
        "а": "a", "в": "b", "с": "c", "ԁ": "d", "е": "e", "ё": "e", "һ": "h", "н": "h", "і": "i", "ї": "i",
        "ј": "j", "к": "k", "м": "m", "о": "o", "р": "p", "ԛ": "q", "ѕ": "s", "т": "t", "у": "y", "х": "x",
        "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
        "υ": "u", "χ": "x", "ω": "w", "ɡ": "g", "ı": "i", "ł": "l", "ø": "o", "đ": "d", "ß": "ss",
    }
)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g",
                       "@": "a", "$": "s", "!": "i", "|": "l", "+": "t"})
_SPLIT = re.compile(r"[^\w@$!|+]+|_+")
_AGE = re.compile(r"^(\d{1,2})(?:yo|yr|yrs|y|year|years|yearold|yearsold)?$")
_AGE_UNITS = {"yo", "y", "yr", "yrs", "year", "years", "yearold", "yearsold", "old"}
_MINOR_TOKEN = "\x00minor"

ABSOLUTE_TERMS = [
    "csam", "cp porn", "child porn", "child pornography", "child sexual", "child nude", "child nudes", "kiddie porn",
    "kiddy porn", "kid porn", "preteen porn", "preteen nude", "preteen sex", "underage porn", "underage sex",
    "underage nude", "pedo porn", "pedophile porn", "jailbait", "lolicon", "shotacon", "toddlercon",
]
_ABSOLUTE_PREFIXES = ["childporn", "kiddieporn", "kiddyporn", "underageporn", "pedoporn", "lolicon", "shotacon"]
MINOR_TERMS = [
    "child", "children", "kid", "kids", "kiddie", "kiddy", "minor", "minors", "underage", "under age", "preteen", "pre teen",
    "preteens", "tween", "tweens", "toddler", "toddlers", "infant", "infants", "schoolgirl", "schoolgirls", "schoolboy",
    "schoolboys", "teen", "teens", "teenage", "teenager", "teenagers", "young girl", "young boy", "little girl",
    "little boy", "loli", "shota", "middle schooler", "elementary schooler", _MINOR_TOKEN,
]
# Kept deliberately narrow: these only block next to a minor term, where over-blocking is the
# acceptable error, but words like "strip", "breasts" or "bottomless" catch too many innocent prompts.
SEXUAL_TERMS = [
    "porn", "porno", "pron", "sex", "sexy", "sexual", "sexually", "nude", "nudes", "naked", "nudity", "nsfw", "explicit",
    "lewd", "hentai", "xxx", "topless", "genitals", "lingerie", "fetish", "seductive", "orgasm", "intercourse",
    "rape", "raped", "grope", "groped", "stripper",
]
_SEXUAL_PREFIXES = ["porn", "erotic", "masturbat", "molest", "undress", "genital", "fornicat"]
CLOTHING_TERMS = ["clothes", "clothing", "clothed", "dressed", "underwear", "swimsuit", "bra", "panties"]
DEEPFAKE_TERMS = ["deepfake", "deep fake", "faceswap", "face swap", "celebrity", "real person"]
COOCCURRENCE_WINDOW = 12
MAX_JOIN = 4


def _collapse(word: str) -> str:
    """Squeezes repeated letters ("chiiild" -> "child") on both sides of a comparison."""
    return re.sub(r"(.)\1+", r"\1", word)


def normalize_tokens(text: str) -> list[str]:
    """Folds text into comparable tokens. Used for matching only; never logged."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = unicodedata.normalize("NFKD", text.casefold().translate(_CONFUSABLES))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens: list[str] = []
    raw = [t for t in _SPLIT.split(text) if t]
    for index, token in enumerate(raw):
        age = _AGE.match(token)
        if age and (token != age.group(1) or (index + 1 < len(raw) and raw[index + 1] in _AGE_UNITS)):
            tokens.append(_MINOR_TOKEN if int(age.group(1)) < 18 else token)
            continue
        if token.isdigit():
            tokens.append(token)
            continue
        folded = "".join(ch for ch in token.translate(_LEET) if ch.isalpha())
        if folded:
            tokens.append(folded)
    return _merge_single_letters(tokens)


def _merge_single_letters(tokens: list[str]) -> list[str]:
    """Joins spelled-out words: "c s a m" and "c.s.a.m" both become "csam"."""
    merged, run = [], []
    for token in tokens + [""]:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if len(run) >= 3:
            merged.append("".join(run))
        else:
            merged.extend(run)
        run = []
        if token:
            merged.append(token)
    return merged


@dataclass(frozen=True)
class _TermSet:
    exact: frozenset[str]
    prefixes: tuple[str, ...]

    @classmethod
    def build(cls, terms: Sequence[str], prefixes: Sequence[str] = ()) -> _TermSet:
        exact: set[str] = set()
        for term in terms:
            joined = term.replace(" ", "")
            exact.add(joined)
            if len(_collapse(joined)) >= 4:
                exact.add(_collapse(joined))
        return cls(frozenset(exact), tuple(p for p in prefixes) + tuple(_collapse(p) for p in prefixes))

    def matches(self, candidate: str) -> bool:
        return candidate in self.exact or _collapse(candidate) in self.exact or candidate.startswith(self.prefixes)


class Blocklist:
    """Deterministic first stage. Extra absolute terms can be added per deployment."""

    def __init__(self, extra_absolute_terms: Sequence[str] = ()):
        self.absolute = _TermSet.build(ABSOLUTE_TERMS + [t.casefold() for t in extra_absolute_terms], _ABSOLUTE_PREFIXES)
        self.minor = _TermSet.build(MINOR_TERMS)
        self.sexual = _TermSet.build(SEXUAL_TERMS, _SEXUAL_PREFIXES)
        self.clothing = _TermSet.build(CLOTHING_TERMS)
        self.deepfake = _TermSet.build(DEEPFAKE_TERMS)

    @staticmethod
    def _hits(tokens: list[str], terms: _TermSet) -> list[int]:
        """Token positions where a term starts, trying joins of up to MAX_JOIN tokens ("ch ild" -> "child")."""
        hits = []
        for start in range(len(tokens)):
            candidate = ""
            for token in tokens[start : start + MAX_JOIN]:
                candidate += token
                if terms.matches(candidate):
                    hits.append(start)
                    break
        return hits

    def check(self, prompt: str, negative_prompt: str | None = None) -> None:
        tokens = normalize_tokens(prompt)
        negative = normalize_tokens(negative_prompt or "")
        if self._hits(tokens, self.absolute) or self._hits(negative, self.absolute):
            raise SafetyViolation("sexual_minors")
        minors = self._hits(tokens, self.minor)
        if not minors:
            sexual = self._hits(tokens, self.sexual)
            if sexual and self._hits(tokens, self.deepfake):
                raise SafetyViolation("sexual_deepfake")
            return
        sexual = self._hits(tokens, self.sexual)
        if any(abs(m - s) <= COOCCURRENCE_WINDOW for m in minors for s in sexual):
            raise SafetyViolation("sexual_minors")
        # A negative prompt steers away from what it lists: "clothing" there pushes toward nudity.
        if self._hits(negative, self.clothing):
            raise SafetyViolation("sexual_minors")


# ---------------------------------------------------------------- stage 2: prompt classifier

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
    baked into the image. Needs the worker's `safety` extra. Not yet run against the real
    weights in this repository: latency and the exact output format must be checked on
    the target CPU before it is relied on.
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
    frames_to_sample: int = 8

    def check_request(self, prompt: str, negative_prompt: str | None = None) -> None:
        self.blocklist.check(prompt, negative_prompt)
        if self.unavailable or (self.classifier is None and self.require_classifier):
            raise SafetyUnavailable()
        if self.classifier is None:
            return
        # Negative prompts list what to avoid, so a classifier would misread them; stage 1 covers them.
        self._judge(self._score(self.classifier.classify, prompt))

    def check_output(self, video: bytes) -> None:
        """Optional output stage: classifies frames sampled from the finished MP4."""
        if self.frame_classifier is None:
            return
        try:
            frames = sample_frames(video, self.frames_to_sample)
        except Exception:
            raise SafetyUnavailable() from None
        if not frames:
            raise SafetyUnavailable()
        self._judge(self._score(self.frame_classifier.classify_frames, frames))

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
        }

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SafetyGate:
        env = os.environ if env is None else env
        kind = env.get("KUNO_SAFETY_CLASSIFIER", "").strip().lower()
        gate = cls(require_classifier=env.get("KUNO_SAFETY_REQUIRE_CLASSIFIER", "") in ("1", "true", "yes"))
        if env.get("KUNO_SAFETY_THRESHOLDS"):
            gate.thresholds = {**DEFAULT_THRESHOLDS, **json.loads(env["KUNO_SAFETY_THRESHOLDS"])}
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


def sample_frames(video: bytes, count: int, size: int = 224) -> list[Any]:
    """Evenly spaced RGB frames, scaled to size x size, decoded with ffmpeg inside the enclave."""
    import numpy as np  # noqa: PLC0415

    from kuno_protocol.mp4 import probe  # noqa: PLC0415

    from .backends.media_tools import ffmpeg_exe  # noqa: PLC0415

    duration = max(probe(video).duration_s, 0.001)
    with tempfile.TemporaryDirectory(prefix="kuno-safety-") as tmp:
        source = Path(tmp) / "in.mp4"
        source.write_bytes(video)
        result = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(source), "-vf",
             f"fps={count / duration:.6f},scale={size}:{size}", "-frames:v", str(count),
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True, timeout=120,
        )
    frame_bytes = size * size * 3
    raw = result.stdout
    return [np.frombuffer(raw[i : i + frame_bytes], dtype=np.uint8).reshape(size, size, 3) for i in range(0, len(raw) - frame_bytes + 1, frame_bytes)]


# ---------------------------------------------------------------- module interface used by worker.py

_gate: SafetyGate | None = None
_lock = threading.Lock()


def default_gate() -> SafetyGate:
    """The process-wide gate, built from the environment on first use (call early to load models at startup)."""
    global _gate
    with _lock:
        if _gate is None:
            _gate = SafetyGate.from_env()
        return _gate


def configure(gate: SafetyGate | None) -> None:
    """Installs a gate (tests, or a worker that builds its own); None rebuilds from the environment."""
    global _gate
    with _lock:
        _gate = gate


def check_request(prompt: str, negative_prompt: str | None = None) -> None:
    default_gate().check_request(prompt, negative_prompt)


def check_output(video: bytes) -> None:
    default_gate().check_output(video)
