"""TAO/USD for USD-denominated pay, from several public sources at once.

One exchange can be wrong, stale or manipulated for a moment, so the rate is the median of
independent sources, and a round fails closed (the validator keeps its previous weights) when
fewer than two fresh sources answer or they disagree by more than a tolerance. A source that
dates its quote (Coinbase's `time`, CoinGecko's `last_updated_at`) is dropped once the quote is
older than `max_age_s`; Kraken's ticker carries no time and counts as observed when fetched.

The gateway has its own oracle for crediting top-ups; validators don't import gateway code, so
this is a separate implementation with the same sources.
"""

from __future__ import annotations

import logging
import re
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

log = logging.getLogger("kuno.validator.prices")

MIN_SOURCES = 2
DEFAULT_TOLERANCE = 0.02
DEFAULT_MAX_AGE_S = 900.0
# A quote dated this far in the future is a broken clock, not a fresh price.
FUTURE_SKEW_S = 120.0


class PriceUnavailable(RuntimeError):
    """No trustworthy TAO/USD rate this round."""


def _iso_time(text: str) -> float:
    # Coinbase sends up to nanoseconds ("2026-09-14T11:10:00.123456789Z"); datetime takes at most microseconds.
    trimmed = re.sub(r"(\.\d{6})\d+", r"\1", text.strip()).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(trimmed)
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()


def _kraken(data: dict) -> tuple[float, float | None]:
    return float(next(iter(data["result"].values()))["c"][0]), None


def _coinbase(data: dict) -> tuple[float, float | None]:
    return float(data["price"]), _iso_time(data["time"]) if data.get("time") else None


def _coingecko(data: dict) -> tuple[float, float | None]:
    entry = data["bittensor"]
    updated = entry.get("last_updated_at")
    return float(entry["usd"]), float(updated) if updated is not None else None


@dataclass(frozen=True)
class Source:
    url: str
    parse: Callable[[dict], tuple[float, float | None]]


SOURCES: dict[str, Source] = {
    "kraken": Source("https://api.kraken.com/0/public/Ticker?pair=TAOUSD", _kraken),
    "coinbase": Source("https://api.exchange.coinbase.com/products/TAO-USD/ticker", _coinbase),
    "coingecko": Source(
        "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd&include_last_updated_at=true", _coingecko
    ),
}


def _fetch_json(url: str) -> dict:
    response = httpx.get(url, timeout=10.0, headers={"user-agent": "kuno-validator"})
    response.raise_for_status()
    return response.json()


@dataclass(frozen=True)
class TaoUsd:
    usd_per_tao: float
    quotes: dict[str, float]
    spread: float
    at: float
    rejected: dict[str, str] = field(default_factory=dict)


class TaoUsdOracle:
    def __init__(
        self,
        sources: Mapping[str, Source] | None = None,
        fetch: Callable[[str], dict] = _fetch_json,
        tolerance: float = DEFAULT_TOLERANCE,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        min_sources: int = MIN_SOURCES,
        clock: Callable[[], float] = time.time,
    ):
        if not 0 < tolerance < 1:
            raise ValueError("the TAO/USD tolerance must be between 0 and 1")
        if max_age_s <= 0:
            raise ValueError("the TAO/USD maximum age must be positive")
        self.sources = dict(SOURCES if sources is None else sources)
        self.fetch, self.clock = fetch, clock
        self.tolerance, self.max_age_s = tolerance, max_age_s
        # At least two: a single source can't be checked against anything.
        self.min_sources = max(MIN_SOURCES, min_sources)

    def quote(self) -> TaoUsd:
        """The median TAO/USD over fresh sources, or PriceUnavailable."""
        now = self.clock()
        quotes: dict[str, float] = {}
        rejected: dict[str, str] = {}
        for name, source in self.sources.items():
            try:
                price, observed = source.parse(self.fetch(source.url))
            except Exception as exc:  # an unreachable or malformed source just doesn't vote
                rejected[name] = f"unavailable ({type(exc).__name__})"
                continue
            if not (price > 0 and price < float("inf")):
                rejected[name] = "non-positive price"
            elif observed is not None and now - observed > self.max_age_s:
                rejected[name] = f"stale ({now - observed:.0f}s old)"
            elif observed is not None and observed - now > FUTURE_SKEW_S:
                rejected[name] = "dated in the future"
            else:
                quotes[name] = price
        if len(quotes) < self.min_sources:
            detail = "; ".join(f"{k} {v}" for k, v in sorted(rejected.items()))
            raise PriceUnavailable(f"only {len(quotes)} fresh TAO/USD source(s), {self.min_sources} needed ({detail})")
        median = statistics.median(quotes.values())
        spread = (max(quotes.values()) - min(quotes.values())) / median
        if spread > self.tolerance:
            listed = ", ".join(f"{k} {v:g}" for k, v in sorted(quotes.items()))
            raise PriceUnavailable(f"TAO/USD sources disagree by {spread:.2%}, more than {self.tolerance:.2%} ({listed})")
        if rejected:
            log.info("TAO/USD from %d source(s); ignored %s", len(quotes), "; ".join(f"{k} {v}" for k, v in sorted(rejected.items())))
        return TaoUsd(float(median), quotes, spread, now, rejected)
