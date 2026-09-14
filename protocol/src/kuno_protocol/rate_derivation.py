"""Fit VCU weights and miner rates from kuno-bench results: `kuno-devkit derive-rates`.

    kuno-devkit derive-rates bench-h200.json bench-b200.json --gpu-price h200=3.20 --gpu-price b200=4.50 \
        --utilization 0.6 --margin 1.25 --write-proposal rates-proposal.json

Units. A VCU is one anchor-GPU-second of cost (--anchor-gpu, default the H200), as profiles.json's weights are
anchored: `h3` at 5 s takes about 60 H200-seconds per output second on 4 × H200, and weighs 60. A cell measured
on another GPU counts its GPU-seconds × its price ÷ the anchor's price, and where several machines measured the
same cell the median counts (research_pricing.md §3, "the median eligible hardware"). GPU-seconds are the
profile's GPUs per worker × the cell's median warm wall time, per output second.

Fit, per profile and resolution, of `VcuWeights` (weight × fps multiplier × (1 + slope × max(0, seconds − 5))):
  weight, slope     least squares over the cells at the default rates (not 48/50 fps) of at least 5 s:
                    VCU per output second = weight + weight × slope × (seconds − 5), the slope clamped at 0
  fps multipliers   for each 48/50 fps, the least-squares ratio of its cells to the fitted line
Clips under 5 s get no discount in VCU, so they are not fitted; the per-cell table shows what they cost against
the fit. profiles.json holds one slope and one multiplier per fps for a whole profile, fitted jointly across its
resolutions. A resolution, slope or multiplier with no measurement keeps its current value and is flagged.

Rates:
  usd_per_vcu_second  confidential: anchor price ÷ 3600 × (1 + --cc-overhead) ÷ --utilization × --margin;
                      open: × --open-tier-share
  gpu_hour_usd        per family: --capacity-share × the cheapest priced GPU that benchmarked the family, so ready
                      capacity alone stays below that GPU's price

Margin check: each (profile, resolution, privacy mode) where some allowed fps and duration prices the job
(`ModelProfile.price_usd`: the per-second rate, fps and long-clip multipliers, and the minimum charge) below the
miner's pay for it (proposed VCU × the confidential rate, the higher tier) × --min-customer-multiple (1.15, §3 item 5).

Writes nothing unless --write-proposal is given. Refuses simulated results (`kuno-bench --backend mock`) unless
--allow-simulated.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .profiles import ModelProfile, ParamError, VcuWeights, load_profiles
from .rate_card import PLACEHOLDER_OPEN_TIER_SHARE, PLACEHOLDER_USD_PER_GPU_HOUR, PLACEHOLDER_USD_PER_VCU_SECOND
from .schemas import GenerationParams
from .tiers import CONFIDENTIAL, OPEN

BENCH_SCHEMA = "kuno-bench"
BENCH_SCHEMA_VERSIONS = (1,)
PROPOSAL_SCHEMA = "kuno-rate-proposal"
PROPOSAL_SCHEMA_VERSION = 1
HIGH_FPS = 48


class RateDerivationError(ValueError):
    """The inputs cannot produce a proposal; the message says why."""


@dataclass(frozen=True)
class Settings:
    gpu_prices: dict[str, float]
    anchor_gpu: str = "h200"
    utilization: float = 0.6
    margin: float = 1.25
    cc_overhead: float = 0.0
    open_tier_share: float = PLACEHOLDER_OPEN_TIER_SHARE
    capacity_share: float = 0.75
    min_customer_multiple: float = 1.15
    allow_simulated: bool = False

    def __post_init__(self) -> None:
        for name in ("utilization", "margin", "open_tier_share", "capacity_share", "min_customer_multiple"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise RateDerivationError(f"--{name.replace('_', '-')} must be a positive number")
        if self.utilization > 1:
            raise RateDerivationError("--utilization is a share of the hour, at most 1")
        if not math.isfinite(self.cc_overhead) or self.cc_overhead < 0:
            raise RateDerivationError("--cc-overhead must be a non-negative share")


def parse_gpu_prices(values: list[str]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for value in values:
        name, sep, price = value.partition("=")
        try:
            usd = float(price)
        except ValueError:
            usd = math.nan
        if not sep or not name.strip() or not math.isfinite(usd) or usd <= 0:
            raise RateDerivationError(f"--gpu-price {value!r}: expected GPU=USD_PER_GPU_HOUR, e.g. h200=3.20")
        prices[name.strip().lower()] = usd
    return prices


def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def price_for(gpu_model: str, prices: dict[str, float]) -> tuple[str, float] | None:
    """The price whose name's words all appear in the GPU model ("rtx-pro-6000" matches "NVIDIA RTX PRO 6000 Blackwell
    Server Edition"); the most specific wins."""
    matches = [(len(_tokens(name)), name, usd) for name, usd in prices.items() if _tokens(name) <= _tokens(gpu_model)]
    if not matches:
        return None
    best = max(size for size, _, _ in matches)
    top = [(name, usd) for size, name, usd in matches if size == best]
    if len(top) > 1:
        raise RateDerivationError(f"{gpu_model}: --gpu-price names {', '.join(n for n, _ in top)} match it equally")
    return top[0]


# ------------------------------------------------------------------ reading bench files


@dataclass
class Sample:
    """One benchmarked cell on one machine, in VCU per output second."""

    profile_id: str
    resolution: str
    fps: int
    duration_s: float
    vcu_per_second: float
    gpu_seconds_per_second: float
    machine: str


@dataclass
class Inputs:
    samples: list[Sample] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    family_prices: dict[str, list[float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    simulated: bool = False


def load_bench(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise RateDerivationError(f"{path}: not a readable JSON file ({exc})") from None
    if not isinstance(doc, dict) or doc.get("schema") != BENCH_SCHEMA:
        raise RateDerivationError(f"{path}: not a kuno-bench results file")
    if doc.get("schema_version") not in BENCH_SCHEMA_VERSIONS:
        raise RateDerivationError(f"{path}: kuno-bench schema_version {doc.get('schema_version')} is not supported here")
    return doc


def collect(benches: list[tuple[str, dict[str, Any]]], settings: Settings, profiles: dict[str, ModelProfile]) -> Inputs:
    anchor = settings.gpu_prices.get(settings.anchor_gpu.lower())
    if anchor is None:
        raise RateDerivationError(f"the anchor GPU {settings.anchor_gpu!r} needs a price: pass --gpu-price {settings.anchor_gpu}=USD_PER_HOUR")
    inputs = Inputs()
    for label, doc in benches:
        machine = doc.get("machine") or {}
        simulated = bool(doc.get("simulated") or machine.get("simulated"))
        if simulated and not settings.allow_simulated:
            raise RateDerivationError(f"{label}: simulated results (kuno-bench --backend mock); pass --allow-simulated to use them anyway")
        inputs.simulated |= simulated
        gpu = machine.get("gpu_model") or ""
        match = price_for(gpu, settings.gpu_prices) if gpu else None
        entry = {"path": label, "gpu_model": gpu or None, "gpu_count": machine.get("gpu_count"), "cc_mode": machine.get("cc_mode"),
                 "simulated": simulated, "price_key": match[0] if match else None, "used_cells": 0}
        inputs.files.append(entry)
        if match is None:
            inputs.warnings.append(f"{label}: no --gpu-price matches {gpu or 'an unknown GPU'}; its cells are ignored")
            continue
        if machine.get("cc_mode") == "on":
            inputs.warnings.append(f"{label}: measured with the GPUs in CC mode; add no --cc-overhead for it")
        price = match[1]
        for record in doc.get("profiles", []):
            profile = profiles.get(record.get("profile"))
            if profile is None:
                inputs.warnings.append(f"{label}: unknown profile {record.get('profile')!r} ignored")
                continue
            gpus = int(record.get("gpus_per_worker") or profile.gpus_per_worker)
            for cell in record.get("cells", []):
                wall = cell.get("wall_s") or {}
                if cell.get("outcome") != "ok" or not wall.get("median") or not cell.get("duration_s"):
                    continue
                gpu_seconds = gpus * float(wall["median"]) / float(cell["duration_s"])
                inputs.samples.append(Sample(profile.id, cell["resolution"], int(cell["fps"]), float(cell["duration_s"]),
                                             gpu_seconds * price / anchor, gpu_seconds, f"{gpu} ({label})"))
                inputs.family_prices.setdefault(profile.family, []).append(price)
                entry["used_cells"] += 1
    if not inputs.samples:
        raise RateDerivationError("no successful, priced benchmark cells in the input files")
    return inputs


# ------------------------------------------------------------------ fitting


def _multiplier_fps(profile: ModelProfile) -> set[int]:
    return set(profile.vcu_weights.fps_multiplier) | {fps for fps in profile.limits.fps if fps >= HIGH_FPS}


def _line(points: list[tuple[float, float]]) -> tuple[float, float | None]:
    """y = a + b·h by least squares, with b ≥ 0; b is None when every h is the same."""
    ys = [y for _, y in points]
    if len({h for h, _ in points}) < 2:
        return statistics.fmean(ys), None
    mean_h, mean_y = statistics.fmean(h for h, _ in points), statistics.fmean(ys)
    b = sum((h - mean_h) * (y - mean_y) for h, y in points) / sum((h - mean_h) ** 2 for h, _ in points)
    if b < 0:
        return mean_y, 0.0
    return mean_y - b * mean_h, b


def _ratio(pairs: list[tuple[float, float]]) -> float | None:
    """The least-squares m in measured ≈ m × predicted."""
    denominator = sum(p * p for _, p in pairs)
    return sum(y * p for y, p in pairs) / denominator if denominator > 0 else None


def _sig(value: float, digits: int = 3) -> float:
    return float(f"{value:.{digits}g}")


def fit_profile(profile: ModelProfile, cells: dict[str, dict[tuple[int, float], float]]) -> dict[str, Any]:
    """`cells`: resolution -> (fps, seconds) -> median VCU per output second."""
    current = profile.vcu_weights
    base_s = current.duration_base_s
    high = _multiplier_fps(profile)
    by_resolution: dict[str, Any] = {}
    weights: dict[str, float] = {}
    sources: dict[str, str] = {}
    notes: list[str] = []
    for resolution in profile.limits.sizes:
        measured = cells.get(resolution, {})
        base = [(max(0.0, d - base_s), y) for (fps, d), y in measured.items() if fps not in high and d >= base_s]
        short_only = False
        if not base:
            base = [(0.0, y) for (fps, d), y in measured.items() if fps not in high]
            short_only = bool(base)
        if not base:
            weights[resolution] = current.per_output_second[resolution]
            sources[resolution] = "current"
            notes.append(f"{profile.id} {resolution}: not benchmarked at the default frame rate; keeps its current weight")
            continue
        if short_only:
            notes.append(f"{profile.id} {resolution}: only clips under {base_s:g} s were measured; the weight includes their fixed costs")
        a, b = _line(base)
        slope = b / a if b is not None and a > 0 else None
        line_slope = slope if slope is not None else current.duration_slope
        multipliers = {}
        for fps in sorted(high):
            pairs = [(y, a * (1 + line_slope * max(0.0, d - base_s))) for (f, d), y in measured.items() if f == fps]
            ratio = _ratio(pairs)
            if ratio is not None:
                multipliers[fps] = round(ratio, 3)
        weights[resolution] = a
        sources[resolution] = "measured"
        by_resolution[resolution] = {"weight": _sig(a), "duration_slope": round(slope, 4) if slope is not None else None,
                                     "fps_multiplier": multipliers, "cells": len(measured), "short_clips_only": short_only}

    fitted = [r for r in weights if sources[r] == "measured"]
    num = sum(h * (y / weights[r] - 1) for r in fitted for (fps, d), y in cells[r].items() if fps not in high and (h := d - base_s) > 0)
    den = sum((d - base_s) ** 2 for r in fitted for (fps, d), _ in cells[r].items() if fps not in high and d > base_s)
    if den > 0:
        slope, slope_source = max(0.0, num / den), "measured"
    else:
        slope, slope_source = current.duration_slope, "current"
        if fitted:
            notes.append(f"{profile.id}: no default-rate clip longer than {base_s:g} s; keeps the current duration slope")
    multipliers: dict[int, float] = {}
    multiplier_sources: dict[int, str] = {}
    for fps in sorted(high):
        pairs = [(y, weights[r] * (1 + slope * max(0.0, d - base_s))) for r in fitted for (f, d), y in cells[r].items() if f == fps]
        ratio = _ratio(pairs)
        if ratio is not None:
            multipliers[fps], multiplier_sources[fps] = round(ratio, 3), "measured"
        elif fps in current.fps_multiplier:
            multipliers[fps], multiplier_sources[fps] = current.fps_multiplier[fps], "current"
            if fitted:
                notes.append(f"{profile.id}: no {fps} fps cell at a resolution fitted at the default rate; keeps its current multiplier")
    proposed = VcuWeights(
        per_output_second={r: (_sig(w) if sources[r] == "measured" else w) for r, w in weights.items()},
        duration_slope=round(slope, 4),
        duration_base_s=base_s,
        fps_multiplier=multipliers,
        note="PROPOSED by kuno-devkit derive-rates from kuno-bench results; review before adopting",
    )
    return {
        "vcu_weights": proposed,
        "by_resolution": by_resolution,
        "sources": {"per_output_second": sources, "duration_slope": slope_source, "fps_multiplier": multiplier_sources},
        "notes": notes,
    }


# ------------------------------------------------------------------ proposal


def _change(current: float | None, proposed: float | None) -> float | None:
    if current is None or proposed is None or current == 0:
        return None
    return round((proposed - current) / current, 3)


def _durations(profile: ModelProfile, fps: int) -> list[float]:
    lim = profile.limits
    longest = min(lim.max_duration_s, lim.max_duration_s_by_fps.get(fps, lim.max_duration_s))
    out, seconds = [], lim.min_duration_s
    while seconds <= longest + 1e-9:
        out.append(round(seconds, 6))
        seconds += lim.duration_step_s
    return out


def margin_check(profiles: dict[str, ModelProfile], weights: dict[str, VcuWeights], usd_per_vcu: float, multiple: float) -> list[dict[str, Any]]:
    rows = []
    for profile in profiles.values():
        proposed = profile.model_copy(update={"vcu_weights": weights[profile.id]})
        for resolution, ratios in profile.limits.sizes.items():
            aspect = next(iter(ratios))
            for privacy in profile.privacy_modes:
                checked, failing = 0, []
                worst: dict[str, Any] | None = None
                for fps in profile.limits.fps:
                    for seconds in _durations(profile, fps):
                        params = GenerationParams(profile_id=profile.id, mode=profile.modes[0], duration_s=seconds, resolution=resolution,
                                                  aspect_ratio=aspect, fps=fps)
                        try:
                            customer = profile.price_usd(params, privacy)
                        except ParamError:
                            continue
                        miner = proposed.vcu_at(resolution, fps, seconds) * usd_per_vcu
                        cell = {"fps": fps, "duration_s": seconds, "customer_usd": round(customer, 4), "miner_usd": round(miner, 4),
                                "ratio": round(customer / miner, 3) if miner > 0 else None}
                        checked += 1
                        if worst is None or (cell["ratio"] or math.inf) < (worst["ratio"] or math.inf):
                            worst = cell
                        if customer < miner * multiple:
                            failing.append(cell)
                if failing:
                    rows.append({"profile": profile.id, "resolution": resolution, "privacy": privacy, "worst": worst,
                                 "failing_cells": len(failing), "checked_cells": checked, "cells": failing})
    return rows


def derive(benches: list[tuple[str, dict[str, Any]]], settings: Settings, profiles: dict[str, ModelProfile] | None = None) -> dict[str, Any]:
    profiles = profiles if profiles is not None else load_profiles()
    inputs = collect(benches, settings, profiles)
    anchor_price = settings.gpu_prices[settings.anchor_gpu.lower()]

    grouped: dict[str, dict[str, dict[tuple[int, float], list[float]]]] = {}
    for sample in inputs.samples:
        grouped.setdefault(sample.profile_id, {}).setdefault(sample.resolution, {}).setdefault((sample.fps, sample.duration_s), []).append(sample.vcu_per_second)
    medians = {p: {r: {k: statistics.median(v) for k, v in cells.items()} for r, cells in res.items()} for p, res in grouped.items()}

    notes = list(inputs.warnings)
    fits: dict[str, Any] = {}
    weights: dict[str, VcuWeights] = {}
    for profile in profiles.values():
        if profile.id not in medians:
            weights[profile.id] = profile.vcu_weights
            fits[profile.id] = {"source": "current", "by_resolution": {}, "sources": {}}
            notes.append(f"{profile.id}: not benchmarked; keeps its current VCU weights")
            continue
        fit = fit_profile(profile, medians[profile.id])
        weights[profile.id] = fit["vcu_weights"]
        fits[profile.id] = {"source": "measured", "by_resolution": fit["by_resolution"], "sources": fit["sources"]}
        notes += fit["notes"]

    confidential = anchor_price / 3600 * (1 + settings.cc_overhead) / settings.utilization * settings.margin
    usd_per_vcu = {CONFIDENTIAL: round(confidential, 8), OPEN: round(confidential * settings.open_tier_share, 8)}
    gpu_hour = {}
    gpu_hour_sources = {}
    for family in sorted({p.family for p in profiles.values()}):
        prices = inputs.family_prices.get(family)
        if prices:
            gpu_hour[family], gpu_hour_sources[family] = round(settings.capacity_share * min(prices), 2), "measured"
        elif family in PLACEHOLDER_USD_PER_GPU_HOUR:
            gpu_hour[family], gpu_hour_sources[family] = PLACEHOLDER_USD_PER_GPU_HOUR[family], "current"
            notes.append(f"{family}: no benchmark priced it; gpu_hour_usd keeps the placeholder")

    cells_table = []
    for profile_id, resolutions in medians.items():
        proposed = weights[profile_id]
        for resolution, cells in sorted(resolutions.items()):
            for (fps, seconds), vcu in sorted(cells.items()):
                fitted = proposed.per_second(resolution, fps, seconds)
                cells_table.append({"profile": profile_id, "resolution": resolution, "fps": fps, "duration_s": seconds,
                                    "measured_vcu_per_second": round(vcu, 3), "fitted_vcu_per_second": round(fitted, 3) if fitted else None,
                                    "measured_over_fitted": round(vcu / fitted, 3) if fitted else None,
                                    "machines": len(grouped[profile_id][resolution][(fps, seconds)])})

    diff = []
    for profile in profiles.values():
        now, new = profile.vcu_weights, weights[profile.id]
        source = fits[profile.id].get("sources", {})
        for resolution in profile.limits.sizes:
            diff.append({"section": "vcu_weights", "profile": profile.id, "key": f"per_output_second.{resolution}",
                         "current": now.per_output_second.get(resolution), "proposed": new.per_output_second.get(resolution),
                         "source": source.get("per_output_second", {}).get(resolution, "current")})
        diff.append({"section": "vcu_weights", "profile": profile.id, "key": "duration_slope", "current": now.duration_slope,
                     "proposed": new.duration_slope, "source": source.get("duration_slope", "current")})
        for fps in sorted(set(now.fps_multiplier) | set(new.fps_multiplier)):
            diff.append({"section": "vcu_weights", "profile": profile.id, "key": f"fps_multiplier.{fps}", "current": now.fps_multiplier.get(fps),
                         "proposed": new.fps_multiplier.get(fps), "source": source.get("fps_multiplier", {}).get(fps, "current")})
    placeholder_vcu = {CONFIDENTIAL: PLACEHOLDER_USD_PER_VCU_SECOND, OPEN: round(PLACEHOLDER_USD_PER_VCU_SECOND * PLACEHOLDER_OPEN_TIER_SHARE, 8)}
    for tier in (CONFIDENTIAL, OPEN):
        diff.append({"section": "rate_card", "key": f"usd_per_vcu_second.{tier}", "current": placeholder_vcu[tier], "proposed": usd_per_vcu[tier],
                     "source": "measured"})
    for family, rate in gpu_hour.items():
        diff.append({"section": "rate_card", "key": f"gpu_hour_usd.{family}", "current": PLACEHOLDER_USD_PER_GPU_HOUR.get(family), "proposed": rate,
                     "source": gpu_hour_sources[family]})
    for row in diff:
        row["change"] = _change(row["current"], row["proposed"])

    return {
        "schema": PROPOSAL_SCHEMA,
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "simulated": inputs.simulated,
        "inputs": {
            "bench_files": inputs.files,
            "gpu_prices_usd_per_hour": dict(settings.gpu_prices),
            "anchor_gpu": settings.anchor_gpu.lower(),
            "utilization": settings.utilization,
            "margin": settings.margin,
            "cc_overhead": settings.cc_overhead,
            "open_tier_share": settings.open_tier_share,
            "capacity_share": settings.capacity_share,
            "min_customer_multiple": settings.min_customer_multiple,
        },
        "vcu_weights": {profile_id: w.model_dump(mode="json") for profile_id, w in weights.items()},
        "fits": fits,
        "cells": cells_table,
        "rate_card": {"usd_per_vcu_second": usd_per_vcu, "gpu_hour_usd": gpu_hour},
        "margin_check": margin_check(profiles, weights, usd_per_vcu[CONFIDENTIAL], settings.min_customer_multiple),
        "diff": diff,
        "notes": notes,
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def render(proposal: dict[str, Any]) -> str:
    inputs = proposal["inputs"]
    lines = []
    if proposal["simulated"]:
        lines.append("SIMULATED INPUT: these numbers come from kuno-bench --backend mock and mean nothing about real GPUs.")
    lines.append(f"Anchor: 1 VCU = one {inputs['anchor_gpu']}-second of cost at ${inputs['gpu_prices_usd_per_hour'][inputs['anchor_gpu']]:.2f}/GPU-hour; "
                 f"utilization {inputs['utilization']:g}, margin {inputs['margin']:g}, CC overhead {inputs['cc_overhead']:g}")
    for entry in inputs["bench_files"]:
        lines.append(f"  {entry['path']}: {entry['gpu_count']} × {entry['gpu_model']} (price {entry['price_key'] or 'none'}), {entry['used_cells']} cells")
    lines += ["", "Diff against profiles.json vcu_weights and rate_card.py placeholders (current -> proposed):"]
    rows = [("section", "profile/tier", "key", "current", "proposed", "change", "source")]
    for row in proposal["diff"]:
        change = f"{row['change']:+.0%}" if row["change"] is not None else "-"
        rows.append((row["section"], row.get("profile", ""), row["key"], _fmt(row["current"]), _fmt(row["proposed"]), change, row["source"]))
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    lines += ["  " + "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)).rstrip() for r in rows]
    lines += ["", f"Margin check: customer price below {inputs['min_customer_multiple']:g} × the confidential miner pay"]
    if not proposal["margin_check"]:
        lines.append("  every profile, resolution and privacy mode clears it")
    for row in proposal["margin_check"]:
        worst = row["worst"]
        lines.append(f"  {row['profile']} {row['resolution']} {row['privacy']}: {row['failing_cells']}/{row['checked_cells']} cells; worst "
                     f"{worst['fps']} fps {worst['duration_s']:g} s: customer ${worst['customer_usd']:.4f} vs miner ${worst['miner_usd']:.4f} "
                     f"(×{worst['ratio']})")
    if proposal["notes"]:
        lines += ["", "Notes:"] + [f"  - {note}" for note in proposal["notes"]]
    return "\n".join(lines)


__all__ = ["RateDerivationError", "Settings", "derive", "load_bench", "parse_gpu_prices", "price_for", "render"]
