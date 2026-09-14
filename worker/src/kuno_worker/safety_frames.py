"""Frame-level output safety: sample the finished video, score the frames, decide.

Prompts and outputs are end-to-end encrypted, so the enclave is the only place a policy can
look at what a model actually rendered. `safety.SafetyGate.check_output` drives this module;
it holds no exceptions of its own so that `safety` can import it without a cycle. Everything
here raises plain exceptions, and the gate turns any of them into `SafetyUnavailable`.

Models (CPU only, the GPUs are busy generating; weights from local directories baked into
the attested image, never downloaded):
  - a sexual-content image classifier (`ImageClassifierFrames`), any Hugging Face
    image-classification checkpoint whose labels map onto `sexual` / `suggestive`:
    Freepik/nsfw_image_detector (MIT, EVA02-base 448px, the recommended one),
    Falconsai/nsfw_image_detection (Apache-2.0, ViT-B/16 224px), or
    Marqo/nsfw-image-detection-384 (Apache-2.0, ViT-tiny 384px, timm);
  - a zero-shot apparent-minor detector (`ZeroShotMinorFrames`) over a CLIP checkpoint,
    openai/clip-vit-large-patch14 (MIT) recommended, openai/clip-vit-base-patch16 lighter.

All sexual content is banned in every privacy mode; nothing in this module can allow it.

Scores are probabilities in [0, 1] per category, one mapping per frame:
  sexual      explicit sexual content or nudity
  suggestive  sexual, or sexualized without explicit nudity (always >= sexual)
  minor       an apparent child or teenager is depicted (photographic or drawn)

Nothing here logs or returns frame data; frames never leave this process.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

POLICY_CATEGORIES = frozenset({"sexual", "suggestive", "minor"})


class FrameScorer(Protocol):
    name: str
    # Square input edge the model wants; the gate samples frames at the largest one configured.
    input_size: int

    def score_frames(self, frames: Sequence[Any]) -> list[dict[str, float]]:
        """One {category: probability} mapping per HxWx3 uint8 RGB frame, in order. May raise."""


@dataclass(frozen=True)
class RequestSignals:
    """What the request-level checks learned, as booleans only: never prompt text."""

    mentions_minor: bool = False


# ---------------------------------------------------------------- policy


@dataclass(frozen=True)
class FramePolicy:
    """Clip-level decision over per-frame scores.

    Every category is reduced to its maximum over all sampled frames, so a minor visible in
    one frame and sexual content in another still block together. That over-blocks some
    legitimate edits (a family scene cut against a mildly suggestive adult clip); it is the
    intended error.

    Defaults and why (none of them has been evaluated for recall; see SECURITY.md):
      sexual 0.4            explicit content or nudity blocks below even odds: sexual content is
                            banned, so the error to prefer is over-blocking.
      suggestive 0.8        clearly sexualised content without nudity (Freepik's "low" class and
                            above) blocks too. Set high because "low" also covers ordinary
                            swimwear and dance clips; the prompt policy catches sexualised intent.
      minor 0.3             CLIP zero-shot mass on child/teen prompts. Deliberately low: a
                            young-looking adult counts as a minor for this rule.
      minor_sexual 0.15     with an apparent minor, a sexual score far below "probably
                            sexual" blocks. Benign synthetic clips peaked at 0.012 (a flat grey
                            frame) on Freepik's model; see scripts/benchmark_frame_safety.py.
      minor_suggestive 0.5  sexualized-but-clothed content with an apparent minor blocks at
                            even odds; lower catches swimwear and dance clips of children.
    Without a minor-presence model, or when the prompt itself mentions a minor, a minor is
    assumed present in every frame, so the minor_* thresholds apply to all content.

    Configuration can only make the policy stricter: an override above the default is refused.
    """

    sexual: float = 0.4
    suggestive: float = 0.8
    minor: float = 0.3
    minor_sexual: float = 0.15
    minor_suggestive: float = 0.5

    THRESHOLD_KEYS = ("sexual", "suggestive", "minor", "minor_sexual", "minor_suggestive")

    def decide(self, rows: Sequence[Mapping[str, float]], signals: RequestSignals | None = None) -> str | None:
        """The violated category ("sexual_minors" or "sexual"), or None to allow."""
        for row in rows:
            for value in row.values():
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("frame classifier returned a score outside [0, 1]")
        sexual = max((row.get("sexual", 0.0) for row in rows), default=0.0)
        suggestive = max((max(row.get("suggestive", 0.0), row.get("sexual", 0.0)) for row in rows), default=0.0)
        minor_rows = [row["minor"] for row in rows if "minor" in row]
        minor = max(minor_rows) if minor_rows else 1.0
        if signals is not None and signals.mentions_minor:
            minor = 1.0
        if minor >= self.minor and (sexual >= self.minor_sexual or suggestive >= self.minor_suggestive):
            return "sexual_minors"
        if sexual >= self.sexual or suggestive >= self.suggestive:
            return "sexual"
        return None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> FramePolicy:
        """KUNO_SAFETY_FRAME_THRESHOLDS={"minor": 0.25, ...}, each at most its default. Bad values raise ValueError."""
        overrides = json.loads(env["KUNO_SAFETY_FRAME_THRESHOLDS"]) if env.get("KUNO_SAFETY_FRAME_THRESHOLDS") else {}
        if not isinstance(overrides, dict):
            raise ValueError("KUNO_SAFETY_FRAME_THRESHOLDS must be a JSON object")
        unknown = set(overrides) - set(cls.THRESHOLD_KEYS)
        if unknown:
            raise ValueError(f"unknown KUNO_SAFETY_FRAME_THRESHOLDS keys: {', '.join(sorted(unknown))}")
        for key, value in overrides.items():
            ceiling = getattr(cls, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= ceiling:
                raise ValueError(f"KUNO_SAFETY_FRAME_THRESHOLDS[{key!r}] must be a number in [0, {ceiling}]: thresholds can only be tightened")
        return cls(**{k: float(v) for k, v in overrides.items()})


# ---------------------------------------------------------------- sampling

# Frames at the end of the clip always decoded, so the true last frame is found even if the
# container's sample count overstates what decodes.
_TAIL = 8


def plan_frame_indices(total: int, count: int) -> list[int]:
    """`count` indices spread evenly over `total` frames, always including the first and last."""
    if total <= 0:
        return []
    if total == 1 or count <= 1:
        return [0] if total == 1 else [0, total - 1]
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def sample_frames(video: bytes, count: int, size: int = 224) -> list[Any]:
    """Evenly spaced RGB frames including the first and last, squashed to size x size.

    Squashing (rather than a center crop) keeps the whole picture in view: a center crop
    of a 16:9 frame drops 44% of its width, where content could sit unexamined.
    Decoded with ffmpeg inside the enclave; raises on anything it cannot decode.
    """
    import numpy as np  # noqa: PLC0415

    from kuno_protocol.mp4 import probe  # noqa: PLC0415

    from .backends.media_tools import ffmpeg_exe  # noqa: PLC0415

    total = probe(video).frames
    planned = plan_frame_indices(total, count)
    if not planned:
        raise ValueError("the video has no frames")
    tail_start = max(0, total - _TAIL)
    head = [i for i in planned if i < tail_start]
    select = "+".join([f"eq(n,{i})" for i in head] + [f"gte(n,{tail_start})"])
    with tempfile.TemporaryDirectory(prefix="kuno-safety-") as tmp:
        source = Path(tmp) / "in.mp4"
        source.write_bytes(video)
        result = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source), "-map", "0:v:0", "-an",
             "-vf", f"select='{select}',scale={size}:{size}", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True, timeout=300,
        )
    frame_bytes = size * size * 3
    raw = result.stdout
    decoded = [np.frombuffer(raw[i : i + frame_bytes], dtype=np.uint8).reshape(size, size, 3)
               for i in range(0, len(raw) - frame_bytes + 1, frame_bytes)]
    if not decoded:
        raise ValueError("no frames decoded")
    # Output order is the head indices, then every frame from tail_start on.
    positions = {p for p in range(len(head))}
    positions.update(len(head) + i - tail_start for i in planned if i >= tail_start)
    positions.add(len(decoded) - 1)
    return [decoded[p] for p in sorted(positions) if p < len(decoded)]


# ---------------------------------------------------------------- adapters

SEXUAL = ("sexual", "suggestive")
# Label names of known checkpoints -> the categories their probability counts toward.
# A category's score is the sum over its labels (softmax outputs, so the sum is a probability).
KNOWN_LABELS: dict[str, tuple[str, ...]] = {
    "nsfw": SEXUAL,      # Falconsai, AdamCodd, Marqo ("NSFW")
    "porn": SEXUAL,
    "hentai": SEXUAL,
    "sexy": ("suggestive",),
    "high": SEXUAL,      # Freepik: explicit
    "medium": SEXUAL,    # Freepik: nudity
    "low": ("suggestive",),
}
SAFE_LABELS = frozenset({"normal", "sfw", "neutral", "safe", "drawings"})


def map_labels(labels: Sequence[str], override: Mapping[str, Any] | None = None) -> list[tuple[str, ...]]:
    """Categories per class index. Unknown labels are an error: a silently ignored class could be the unsafe one."""
    table = {k.lower(): tuple(v) if isinstance(v, (list, tuple)) else (v,) for k, v in (override or {}).items()}
    mapped: list[tuple[str, ...]] = []
    for label in labels:
        key = str(label).lower()
        if key in table:
            mapped.append(tuple(c for c in table[key] if c))
        elif key in KNOWN_LABELS:
            mapped.append(KNOWN_LABELS[key])
        elif key in SAFE_LABELS:
            mapped.append(())
        else:
            raise ValueError(f"frame classifier label {label!r} has no category; set KUNO_SAFETY_FRAME_LABEL_MAP")
    if not any("sexual" in cats for cats in mapped):
        raise ValueError("no frame classifier label maps to the sexual category")
    return mapped


def _pixels(frames: Sequence[Any], size: int, mean: Sequence[float], std: Sequence[float], dtype: Any) -> Any:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415, N812

    batch = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255.0)
    if batch.shape[-1] != size or batch.shape[-2] != size:
        batch = F.interpolate(batch, size=(size, size), mode="bicubic", antialias=True, align_corners=False).clamp_(0.0, 1.0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return ((batch - mean_t) / std_t).to(dtype)


def _torch_dtype(name: str) -> Any:
    import torch  # noqa: PLC0415

    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if name not in dtypes:
        raise ValueError(f"KUNO_SAFETY_FRAME_DTYPE must be one of {', '.join(dtypes)}")
    return dtypes[name]


def _local_dir(path: Path) -> Path:
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError("frame model path must be a local directory with a config.json")
    return path


class ImageClassifierFrames:
    """A sexual-content image classifier from a local Hugging Face checkpoint.

    Transformers checkpoints (`model_type` in config.json: ViT, timm_wrapper) load through
    AutoModelForImageClassification; bare timm checkpoints (`architecture`, e.g. Marqo's) load
    through timm. Freepik's EVA02 and Marqo need the `timm` package.
    """

    name = "image-classifier"

    def __init__(self, model_path: Path, label_map: Mapping[str, Any] | None = None, dtype: str = "float32"):
        path = _local_dir(Path(model_path))
        config = json.loads((path / "config.json").read_text())
        self._dtype = _torch_dtype(dtype)
        pretrained = config.get("pretrained_cfg") or {}
        if "model_type" in config:
            from transformers import AutoModelForImageClassification  # noqa: PLC0415 — optional extra

            self._model = AutoModelForImageClassification.from_pretrained(path, local_files_only=True)
            id2label = self._model.config.id2label
            labels = [id2label[i] for i in range(len(id2label))]
            self._transformers = True
        elif "architecture" in config:
            import timm  # noqa: PLC0415 — optional

            labels = list(config["label_names"])
            weights = path / "model.safetensors"
            self._model = timm.create_model(
                config["architecture"], pretrained=False, num_classes=len(labels), checkpoint_path=str(weights)
            )
            self._transformers = False
        else:
            raise ValueError("config.json names neither a transformers model_type nor a timm architecture")
        processor = path / "preprocessor_config.json"
        pre = json.loads(processor.read_text()) if processor.is_file() else {}
        size = pre.get("size")
        if isinstance(size, dict):
            size = size.get("height") or size.get("shortest_edge")
        self.input_size = int(size or (pretrained.get("input_size") or [config.get("image_size", 224)])[-1])
        self._mean = pre.get("image_mean") or pretrained.get("mean") or [0.5, 0.5, 0.5]
        self._std = pre.get("image_std") or pretrained.get("std") or [0.5, 0.5, 0.5]
        self._categories = map_labels(labels, label_map)
        self._model = self._model.to(self._dtype).eval()
        self.name = f"image-classifier:{path.name}"

    def score_frames(self, frames: Sequence[Any]) -> list[dict[str, float]]:
        import torch  # noqa: PLC0415

        pixels = _pixels(frames, self.input_size, self._mean, self._std, self._dtype)
        with torch.inference_mode():
            out = self._model(pixel_values=pixels) if self._transformers else self._model(pixels)
            logits = out.logits if hasattr(out, "logits") else out
            probs = torch.softmax(logits.float(), dim=-1).tolist()
        rows = []
        for frame_probs in probs:
            row = {"sexual": 0.0, "suggestive": 0.0}
            for categories, p in zip(self._categories, frame_probs):
                for category in categories:
                    row[category] = row.get(category, 0.0) + p
            rows.append({k: min(max(v, 0.0), 1.0) for k, v in row.items()})
        return rows


# Zero-shot prompts. Softmax runs over all three groups, so frames with no person put their
# mass on the scene prompts; `minor` is the mass on the first group. Drawn and animated
# children count: the policy covers synthetic and drawn depictions.
MINOR_PROMPTS = [
    "a photo of a child", "a photo of a young child", "a photo of a little girl", "a photo of a little boy",
    "a photo of a baby", "a photo of a toddler", "a photo of a teenager", "a photo of a teenage girl",
    "a photo of a teenage boy", "a photo of a schoolchild", "a drawing of a child", "an anime illustration of a young girl",
    "a 3d render of a child",
]
ADULT_PROMPTS = [
    "a photo of an adult woman", "a photo of an adult man", "a photo of a middle-aged person",
    "a photo of an elderly person", "a photo of a young adult woman", "a photo of a young adult man",
    "a drawing of an adult", "an anime illustration of an adult woman", "a 3d render of an adult",
]
SCENE_PROMPTS = [
    "a photo of a landscape", "a photo of an animal", "a photo of a building", "a photo of an object",
    "a photo of food", "a photo of a vehicle", "an abstract colorful pattern", "a test pattern", "text on a screen",
    "a dark empty frame",
]


class ZeroShotMinorFrames:
    """Apparent-minor presence from a local CLIP checkpoint, zero-shot.

    The text tower runs once at load to embed the prompts and is then dropped. OpenAI's CLIP
    model card calls deployed use out of scope without in-domain testing: this is one input
    to a block decision that errs toward blocking, and must be evaluated before it is trusted.
    """

    name = "zero-shot-minor"

    def __init__(self, model_path: Path, dtype: str = "float32"):
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415 — optional extra

        path = _local_dir(Path(model_path))
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "clip":
            raise ValueError("the minor-presence model must be a CLIP checkpoint")
        self._dtype = _torch_dtype(dtype)
        model = AutoModel.from_pretrained(path, local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        prompts = MINOR_PROMPTS + ADULT_PROMPTS + SCENE_PROMPTS
        with torch.inference_mode():
            text = _embedding(model.get_text_features(**tokenizer(prompts, padding=True, return_tensors="pt")))
        self._text = torch.nn.functional.normalize(text.float(), dim=-1)
        self._scale = model.logit_scale.detach().exp().item()
        self._minors = len(MINOR_PROMPTS)
        processor = path / "preprocessor_config.json"
        pre = json.loads(processor.read_text()) if processor.is_file() else {}
        self._mean = pre.get("image_mean") or [0.48145466, 0.4578275, 0.40821073]
        self._std = pre.get("image_std") or [0.26862954, 0.26130258, 0.27577711]
        self.input_size = int(model.config.vision_config.image_size)
        self._vision = model.vision_model.to(self._dtype)
        self._projection = model.visual_projection.to(self._dtype)
        del model  # the text tower is no longer needed
        self.name = f"zero-shot-minor:{path.name}"

    def score_frames(self, frames: Sequence[Any]) -> list[dict[str, float]]:
        import torch  # noqa: PLC0415

        pixels = _pixels(frames, self.input_size, self._mean, self._std, self._dtype)
        with torch.inference_mode():
            pooled = self._vision(pixel_values=pixels).pooler_output
            image = torch.nn.functional.normalize(self._projection(pooled).float(), dim=-1)
            probs = torch.softmax(self._scale * image @ self._text.T, dim=-1)
            minor = probs[:, : self._minors].sum(dim=-1).clamp(0.0, 1.0).tolist()
        return [{"minor": m} for m in minor]


def _embedding(output: Any) -> Any:
    """get_*_features returns a tensor in transformers 4 and a model output in some 5.x releases."""
    return output if hasattr(output, "shape") else output.pooler_output


def load_frame_models(env: Mapping[str, str]) -> tuple[list[FrameScorer], list[str]]:
    """Models named by the environment, plus operator warnings. Raises when a configured model cannot load.

    KUNO_SAFETY_FRAME_MODEL_PATH   sexual-content image classifier directory
    KUNO_SAFETY_MINOR_MODEL_PATH   CLIP directory for apparent-minor presence
    KUNO_SAFETY_FRAME_LABEL_MAP    JSON {label: category or [categories]} for unfamiliar classifiers
    KUNO_SAFETY_FRAME_DTYPE        float32 (default) | bfloat16 (fast on CPUs with AMX or AVX512-BF16)
    KUNO_SAFETY_THREADS            torch CPU threads for classification (default: torch's choice)
    """
    frame_path = env.get("KUNO_SAFETY_FRAME_MODEL_PATH", "").strip()
    minor_path = env.get("KUNO_SAFETY_MINOR_MODEL_PATH", "").strip()
    if not frame_path and not minor_path:
        return [], []
    if not frame_path:
        raise ValueError("KUNO_SAFETY_MINOR_MODEL_PATH is set without KUNO_SAFETY_FRAME_MODEL_PATH; it cannot judge sexual content")
    dtype = env.get("KUNO_SAFETY_FRAME_DTYPE", "float32").strip() or "float32"
    if env.get("KUNO_SAFETY_THREADS"):
        import torch  # noqa: PLC0415

        torch.set_num_threads(int(env["KUNO_SAFETY_THREADS"]))
    label_map = json.loads(env["KUNO_SAFETY_FRAME_LABEL_MAP"]) if env.get("KUNO_SAFETY_FRAME_LABEL_MAP") else None
    models: list[FrameScorer] = [ImageClassifierFrames(Path(frame_path), label_map, dtype)]
    warnings = []
    if minor_path:
        models.append(ZeroShotMinorFrames(Path(minor_path), dtype))
    else:
        warnings.append("KUNO_SAFETY_MINOR_MODEL_PATH is not set: every frame is treated as showing a minor, so low sexual scores block")
    return models, warnings
