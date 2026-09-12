"""Media type sniffing and which types each input role accepts."""

from __future__ import annotations

from .profiles import InputRole

IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
VIDEO_TYPES = frozenset({"video/mp4", "video/quicktime", "video/webm"})
AUDIO_TYPES = frozenset({"audio/wav", "audio/mpeg", "audio/ogg", "audio/flac"})

ROLE_TYPES: dict[InputRole, frozenset[str]] = {
    InputRole.FIRST_FRAME: IMAGE_TYPES,
    InputRole.LAST_FRAME: IMAGE_TYPES,
    InputRole.KEYFRAME: IMAGE_TYPES,
    InputRole.REFERENCE_IMAGE: IMAGE_TYPES,
    InputRole.REFERENCE_VIDEO: VIDEO_TYPES,
    InputRole.REFERENCE_AUDIO: AUDIO_TYPES,
    InputRole.SOURCE_VIDEO: VIDEO_TYPES,
}

EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
}


def sniff_mime(data: bytes) -> str | None:
    head = data[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio/wav"
    if head[4:8] == b"ftyp":
        return "video/quicktime" if head[8:10] == b"qt" else "video/mp4"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if head.startswith(b"OggS"):
        return "audio/ogg"
    if head.startswith(b"fLaC"):
        return "audio/flac"
    return None
