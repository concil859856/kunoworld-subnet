"""Canary prompts. Validators generate these as ordinary encrypted jobs, so
miners cannot tell them apart from customer traffic.

Production validators should keep a private, rotating prompt set drawn from the
same distribution as real requests; this public list is only a fallback.
"""

from __future__ import annotations

import random

from kuno_protocol.prompts import NEUTRAL_PROMPTS

# The worker's determinism check builds the same golden cases from the same list, so it lives in the protocol.
FALLBACK_PROMPTS = NEUTRAL_PROMPTS


def pick_prompt(rng: random.Random | None = None) -> str:
    return (rng or random).choice(FALLBACK_PROMPTS)
