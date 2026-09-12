"""Canary prompts. Validators generate these as ordinary encrypted jobs, so
miners cannot tell them apart from customer traffic.

Production validators should keep a private, rotating prompt set drawn from the
same distribution as real requests; this public list is only a fallback.
"""

from __future__ import annotations

import random

FALLBACK_PROMPTS = [
    "A fishing boat returns to harbor at golden hour, gulls circling, gentle waves",
    "Close-up of rain running down a café window, neon signs blurred behind it",
    "A potter's hands shape wet clay on a spinning wheel, soft studio light",
    "Aerial shot over autumn forest with a winding river, morning mist",
    "A street musician plays violin under a stone archway, passersby slow down",
    "Macro shot of a hummingbird hovering near red flowers, shallow depth of field",
    "A night train crosses a snowy bridge, warm lights in the carriages",
    "Chef plating a dessert with precise tweezers, overhead camera",
]


def pick_prompt(rng: random.Random | None = None) -> str:
    return (rng or random).choice(FALLBACK_PROMPTS)
