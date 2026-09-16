"""Neutral prompts the subnet generates with when nobody is asking for anything in particular.

Canary jobs, golden sets and the determinism check all render these: they pass the shared content policy, they
are dull enough to publish, and every component picks them from here so that a case computed on one machine is
the same case on the next.
"""

from __future__ import annotations

NEUTRAL_PROMPTS = [
    "A fishing boat returns to harbor at golden hour, gulls circling, gentle waves",
    "Close-up of rain running down a café window, neon signs blurred behind it",
    "A potter's hands shape wet clay on a spinning wheel, soft studio light",
    "Aerial shot over autumn forest with a winding river, morning mist",
    "A street musician plays violin under a stone archway, passersby slow down",
    "Macro shot of a hummingbird hovering near red flowers, shallow depth of field",
    "A night train crosses a snowy bridge, warm lights in the carriages",
    "Chef plating a dessert with precise tweezers, overhead camera",
]
