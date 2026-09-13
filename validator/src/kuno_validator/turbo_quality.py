"""Quality metrics for the Turbo track: does a faster pipeline still make the video it was asked for?

There is no ground truth for a generated video, so the floor is prompt alignment: an embedding of
the prompt compared with an embedding of the video, on a [0, 1] scale. Every validator must run the
metric the owner-signed spec names, or their floors disagree.

  dev-caption  deterministic, dependency-free; reads a caption the mock backend (or a test) embeds
               in the MP4 and compares hashed bags of words. Development networks and tests only.
  clip         mean CLIP image embedding over sampled frames vs the CLIP text embedding
               (optional deps: torch, transformers, av, pillow)
  xclip        X-CLIP video embedding of sampled frames vs its text embedding (same deps)
  vlm-judge    an OpenAI-compatible vision model scores frames against the prompt (av, pillow, and an
               endpoint the validator trusts with its hidden prompts; they are revealed later anyway)

The model-based metrics are written but not exercised in this repository's tests.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

DEV_CAPTION_MARKER = b"kuno-dev-caption:"
_DEV_CAPTION_MAX = 1024


class QualityMetric(Protocol):
    name: str

    def score(self, prompt: str, video: bytes) -> float:
        """Alignment of `video` with `prompt`, in [0, 1]."""


class Embedder(Protocol):
    def embed_text(self, text: str) -> Sequence[float]: ...

    def embed_video(self, video: bytes) -> Sequence[float] | None:
        """None when the video cannot be embedded (undecodable): it scores zero."""


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class AlignmentMetric:
    """Cosine similarity rescaled to [0, 1] between `low` and `high` (CLIP-family cosines for a matching
    pair sit around 0.25-0.35, so raw cosines are calibrated per metric in the spec's floor values)."""

    def __init__(self, name: str, embedder: Embedder, low: float = 0.0, high: float = 1.0):
        if high <= low:
            raise ValueError("high must exceed low")
        self.name, self.embedder, self.low, self.high = name, embedder, low, high

    def score(self, prompt: str, video: bytes) -> float:
        embedding = self.embedder.embed_video(video)
        if embedding is None:
            return 0.0
        value = (cosine(self.embedder.embed_text(prompt), embedding) - self.low) / (self.high - self.low)
        return min(1.0, max(0.0, value))


# ---------------------------------------------------------------- deterministic dev metric


def hashed_bag_of_words(text: str, dim: int = 256) -> list[float]:
    vector = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        vector[int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % dim] += 1.0
    return vector


def dev_caption_box(caption: str) -> bytes:
    """An MP4 `free` box carrying a caption; appended to a file it keeps the MP4 valid."""
    payload = DEV_CAPTION_MARKER + caption.encode()[:_DEV_CAPTION_MAX] + b"\n"
    return (8 + len(payload)).to_bytes(4, "big") + b"free" + payload


def dev_caption(video: bytes) -> str | None:
    start = video.rfind(DEV_CAPTION_MARKER)
    if start < 0:
        return None
    body = video[start + len(DEV_CAPTION_MARKER) : start + len(DEV_CAPTION_MARKER) + _DEV_CAPTION_MAX]
    return body.split(b"\n", 1)[0].decode("utf-8", "replace")


class DevCaptionEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_text(self, text: str) -> list[float]:
        return hashed_bag_of_words(text, self.dim)

    def embed_video(self, video: bytes) -> list[float] | None:
        caption = dev_caption(video)
        return None if caption is None else hashed_bag_of_words(caption, self.dim)


# ---------------------------------------------------------------- model-based metrics (optional deps)


def sample_frames(video: bytes, count: int) -> list[Any]:
    """`count` RGB PIL images spread evenly over the video. Needs PyAV and Pillow."""
    import av  # noqa: PLC0415 — optional dependency

    with av.open(io.BytesIO(video)) as container:
        frames = [frame.to_image() for frame in container.decode(video=0)]
    if not frames:
        raise ValueError("video has no decodable frames")
    step = max(1, len(frames) // count)
    return frames[::step][:count]


class ClipEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", frames: int = 8, device: str | None = None):
        import torch  # noqa: PLC0415
        from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.frames = frames

    def embed_text(self, text: str) -> list[float]:
        with self._torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            features = self.model.get_text_features(**inputs)
        return self._torch.nn.functional.normalize(features, dim=-1)[0].tolist()

    def embed_video(self, video: bytes) -> list[float] | None:
        try:
            images = sample_frames(video, self.frames)
        except Exception:
            return None
        with self._torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            features = self._torch.nn.functional.normalize(self.model.get_image_features(**inputs), dim=-1)
        return features.mean(dim=0).tolist()


class XClipEmbedder:
    def __init__(self, model_name: str = "microsoft/xclip-base-patch32", frames: int = 8, device: str | None = None):
        import torch  # noqa: PLC0415
        from transformers import XCLIPModel, XCLIPProcessor  # noqa: PLC0415

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = XCLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = XCLIPProcessor.from_pretrained(model_name)
        self.frames = frames

    def embed_text(self, text: str) -> list[float]:
        with self._torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            features = self.model.get_text_features(**inputs)
        return self._torch.nn.functional.normalize(features, dim=-1)[0].tolist()

    def embed_video(self, video: bytes) -> list[float] | None:
        try:
            images = sample_frames(video, self.frames)
        except Exception:
            return None
        while len(images) < self.frames:
            images.append(images[-1])
        with self._torch.no_grad():
            inputs = self.processor(videos=[images], return_tensors="pt").to(self.device)
            features = self.model.get_video_features(**inputs)
        return self._torch.nn.functional.normalize(features, dim=-1)[0].tolist()


class VlmJudgeMetric:
    """Asks a vision-language model for a 0-10 prompt-adherence score over sampled frames.

    Deterministic only as far as the endpoint is (temperature 0, fixed model revision); use it
    with a relative floor (`max_drop_vs_reference`) rather than an absolute one.
    """

    name = "vlm-judge"
    PROMPT = (
        "You grade AI-generated video against its prompt. The images are frames in order. "
        "Score prompt adherence and visual quality together from 0 (unrelated or broken) to 10 "
        '(fully matches, no artifacts). Reply with JSON only: {"score": <number>}.\n\nPrompt: '
    )

    def __init__(self, base_url: str, model: str, api_key: str | None = None, frames: int = 6, timeout: float = 120.0):
        import httpx  # noqa: PLC0415 — already a validator dependency

        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self.model, self.frames = model, frames

    def score(self, prompt: str, video: bytes) -> float:
        try:
            images = sample_frames(video, self.frames)
        except Exception:
            return 0.0
        content: list[dict] = [{"type": "text", "text": self.PROMPT + prompt}]
        for image in images:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=85)
            url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": url}})
        response = self._http.post(
            "/chat/completions",
            json={"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": content}]},
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", text, re.S)
        try:
            value = float(json.loads(match.group(0))["score"]) if match else float(text.strip())
        except (ValueError, KeyError, TypeError):
            return 0.0
        return min(1.0, max(0.0, value / 10.0))


def build_metric(name: str, **options: Any) -> QualityMetric:
    """The metric a spec names. Model metrics import their optional dependencies here."""
    if name == "dev-caption":
        return AlignmentMetric(name, DevCaptionEmbedder(int(options.get("dim", 256))))
    if name == "clip":
        embedder = ClipEmbedder(options.get("model", "openai/clip-vit-large-patch14"), int(options.get("frames", 8)))
        return AlignmentMetric(name, embedder, float(options.get("low", 0.15)), float(options.get("high", 0.35)))
    if name == "xclip":
        embedder = XClipEmbedder(options.get("model", "microsoft/xclip-base-patch32"), int(options.get("frames", 8)))
        return AlignmentMetric(name, embedder, float(options.get("low", 0.15)), float(options.get("high", 0.35)))
    if name == "vlm-judge":
        return VlmJudgeMetric(options["base_url"], options["model"], options.get("api_key"), int(options.get("frames", 6)))
    raise ValueError(f"unknown quality metric {name!r}")
