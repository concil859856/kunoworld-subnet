"""Content safeguards that run inside the enclave.

Nobody outside the VM can see prompts or outputs, so the safety filter has to
live here, baked into every attested image. The MiniMax H3 license also requires
hosted services to maintain safeguards against its Acceptable Use Policy.

This is a minimal pattern gate. Before launch it must be replaced by a prompt
classifier plus output frame/audio classifiers shipped inside the image.
"""

from __future__ import annotations

import re


class SafetyViolation(Exception):
    pass


_BLOCKED = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcsam\b",
        r"\bchild\s+(porn|sexual|nude)",
        r"\b(minor|underage)\b.{0,40}\b(sex|nude|explicit)",
    )
]


def check_request(prompt: str, negative_prompt: str | None = None) -> None:
    for text in (prompt, negative_prompt or ""):
        if any(p.search(text) for p in _BLOCKED):
            raise SafetyViolation("request violates the acceptable use policy")
