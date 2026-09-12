"""Canonical encodings shared by every component.

Anything that is hashed, signed or used as AEAD associated data must be encoded
with these helpers so that Python and TypeScript produce identical bytes.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, no whitespace, UTF-8. Mirrors `canonicalJson` in the JS SDK.

    Integral floats are written as integers (5.0 -> 5) because JavaScript cannot tell
    them apart; other floats use the shortest round-trip form in both languages.
    """
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _normalize(obj: Any) -> Any:
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError("canonical JSON cannot encode NaN or infinity")
        return int(obj) if obj.is_integer() else obj
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def b64e(data: bytes) -> str:
    """URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
