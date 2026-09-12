"""Territory rules imposed by model licenses.

MiniMax H3 Community License (2026-08-02): "Excluded Territories" are the European
Union, the United Kingdom, the Republic of Korea and the United States. Use,
display or distribution of H3 or its Outputs there needs written authorization
from MiniMax. Unknown locations are treated as excluded (conservative safeguard).
"""

from __future__ import annotations

EU27 = frozenset(
    "AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split()
)

REGION_POLICIES: dict[str, frozenset[str]] = {
    "minimax-h3": EU27 | {"US", "GB", "KR"},
}

# Cloudflare uses XX for unknown and T1 for Tor exit nodes.
_UNKNOWN = {"", "XX", "T1", "ZZ"}


def normalize_country(code: str | None) -> str | None:
    if not code:
        return None
    code = code.strip().upper()
    if code in _UNKNOWN:
        return None
    return "GB" if code == "UK" else code


def is_excluded(policy: str | None, country: str | None) -> bool:
    """True when `policy` forbids serving a user located in `country`."""
    if not policy:
        return False
    excluded = REGION_POLICIES.get(policy)
    if excluded is None:
        raise ValueError(f"unknown region policy {policy!r}")
    normalized = normalize_country(country)
    return normalized is None or normalized in excluded
