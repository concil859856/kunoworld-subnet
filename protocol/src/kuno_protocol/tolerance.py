"""Tolerance-mode step replay: accepting a re-executed denoising step within a calibrated distance.

Bitwise replay (verified.py) needs the miner and the validator on one reproducibility domain.
Open-tier hardware (GeForce RTX 4090/5090, workstation RTX PRO 6000, H100 without confidential
computing) cannot share a domain with the validator's GPU, so its hardware classes compare a
replayed step within a tolerance instead (VERIFIED_MODE.md, "Tolerance mode").

The metric (METRIC, "relative update error"):

    Δ     = x_k − x_{k−1}                  the committed step's update
    e     = x̂_k − x_k                      replay minus commitment (x̂_k = step(x_{k−1}) on the validator)
    rel_l2      = ‖e‖₂  / max(‖Δ‖₂,  FLOOR · ‖x_k‖₂)
    max_abs_rel = max|e| / max(max|Δ|, FLOOR · max|x_k|)

computed per tensor (video, audio) in float64 and aggregated by taking the maximum.

Why relative to the update and not to the latent: both sides start from the same committed
x_{k−1}, so the latent's own norm is shared and says nothing. Honest cross-hardware drift is a
small fraction of what one step computes; a substituted model, a skipped step or a cheaper
quantization changes the update itself, by a fraction of order one. Dividing by ‖Δ‖ makes the
two separable with one threshold across early (large dσ) and late (small dσ) steps, where a
latent-relative error would shrink every late-step cheat towards the noise floor. The L2 term
catches diffuse changes; the max-abs term catches a localized edit (a patch of frames) that L2
averages away. Cosine similarity is not used: it ignores magnitude, so a scaled update passes.

Thresholds are never guessed. Each (profile, miner hardware class, executor class) needs an
entry in the calibration file, measured on real GPUs from honest replays (and ideally from
known substitutions); until one exists the audit concludes `unproven`, which never costs the
miner. See `kuno_validator.calibrate` for the tool that turns measured samples into entries.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .verified import DTYPE_SIZES, Tensor, VerifiedModeError

METRIC = "kuno/v1/step-update-rel-l2"
# Guards the denominator when a step barely moves the latent (dσ ≈ 0).
FLOOR = 1e-6
# Wildcard executor class in a calibration entry.
ANY_EXECUTOR = "*"
CALIBRATION_FILE = "tolerance_calibration.json"

BITWISE = "bitwise"
TOLERANCE = "tolerance"


class CalibrationError(ValueError):
    """Samples that cannot yield a trustworthy threshold (too few, or honest and cheating runs overlap)."""


# ------------------------------------------------------------------ distance


def _np():
    import numpy as np

    return np


def tensor_values(tensor: Tensor):
    """A tensor's elements as a flat float64 array. Decodes bfloat16, which numpy has no dtype for."""
    np = _np()
    spec, data = tensor
    if len(data) != spec.nbytes:
        raise VerifiedModeError(f"tensor {spec.name}: {len(data)} bytes for {spec.nbytes}-byte shape")
    if spec.dtype == "bfloat16":
        # 1 sign bit, 8 exponent bits (bias 127, as float32), 7 mantissa bits.
        raw = np.frombuffer(data, dtype="<u2").astype(np.int64)
        sign = np.where(raw >> 15, -1.0, 1.0)
        exponent = (raw >> 7) & 0xFF
        mantissa = (raw & 0x7F).astype(np.float64) / 128.0
        with np.errstate(over="ignore"):
            value = np.where(exponent == 0, np.ldexp(mantissa, -126), np.ldexp(1.0 + mantissa, (exponent - 127).astype(np.int32)))
        value = np.where(exponent == 0xFF, np.where(mantissa == 0, np.inf, np.nan), value)
        return sign * value
    if spec.dtype not in DTYPE_SIZES:
        raise VerifiedModeError(f"unsupported latent dtype {spec.dtype}")
    return np.frombuffer(data, dtype=np.dtype(spec.dtype).newbyteorder("<")).astype(np.float64)


@dataclass(frozen=True)
class StepDistance:
    rel_l2: float
    max_abs_rel: float
    update_l2: float
    # False when the committed latents hold NaN or infinity: never an honest trajectory.
    finite: bool = True

    def record(self, step: int) -> dict:
        """One calibration sample line (JSON Lines), as `kuno_validator.calibrate` reads it."""
        return {"step": step, "rel_l2": self.rel_l2, "max_abs_rel": self.max_abs_rel}


def _by_name(tensors: Sequence[Tensor]) -> dict[str, Tensor]:
    return {spec.name: (spec, data) for spec, data in tensors}


def step_distance(previous: Sequence[Tensor], committed: Sequence[Tensor], replayed: Sequence[Tensor]) -> StepDistance:
    """Distance between a replayed step and the committed one (see the module docstring).

    Raises VerifiedModeError when the replay's layout differs from the commitment's, or the
    replay itself is not finite: those are executor problems, not evidence against the miner.
    """
    np = _np()
    before, after, mine = _by_name(previous), _by_name(committed), _by_name(replayed)
    if set(after) != set(mine) or set(before) != set(after):
        raise VerifiedModeError("replayed latent state names different tensors than the commitment")
    rel_l2 = max_abs_rel = update_l2 = 0.0
    finite = True
    for name in sorted(after):
        specs = (before[name][0], after[name][0], mine[name][0])
        if len({tuple(s.shape) for s in specs}) != 1:
            raise VerifiedModeError(f"tensor {name}: replayed shape differs from the commitment")
        x_prev, x_next, x_hat = (tensor_values(t) for t in (before[name], after[name], mine[name]))
        if not np.all(np.isfinite(x_hat)):
            raise VerifiedModeError(f"tensor {name}: the replay produced non-finite values")
        if not (np.all(np.isfinite(x_prev)) and np.all(np.isfinite(x_next))):
            finite = False
            continue
        update, error = x_next - x_prev, x_hat - x_next
        norm = float(np.linalg.norm(update))
        l2_floor = FLOOR * float(np.linalg.norm(x_next))
        peak = float(np.max(np.abs(update))) if update.size else 0.0
        abs_floor = FLOOR * (float(np.max(np.abs(x_next))) if x_next.size else 0.0)
        rel_l2 = max(rel_l2, _ratio(float(np.linalg.norm(error)), max(norm, l2_floor)))
        max_abs_rel = max(max_abs_rel, _ratio(float(np.max(np.abs(error))) if error.size else 0.0, max(peak, abs_floor)))
        update_l2 = max(update_l2, norm)
    return StepDistance(rel_l2, max_abs_rel, update_l2, finite)


def _ratio(numerator: float, denominator: float) -> float:
    if numerator == 0.0:
        return 0.0
    return numerator / denominator if denominator > 0.0 else math.inf


# ------------------------------------------------------------------ calibration


class DistanceStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: int = Field(ge=1)
    mean: float
    p50: float
    p99: float
    p999: float
    max: float


class CalibrationEntry(BaseModel):
    """One calibrated comparison: miner `hardware_class` replayed on `executor_class` for `profile_id`."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    hardware_class: str
    executor_class: str = ANY_EXECUTOR
    metric: Literal["kuno/v1/step-update-rel-l2"] = METRIC
    # rel_l2 of honest replays (and max_abs_rel), measured on real hardware.
    honest: DistanceStats
    honest_max_abs: DistanceStats | None = None
    # rel_l2 of deliberately substituted steps (another checkpoint, a coarser quantization), when measured.
    substituted: DistanceStats | None = None
    threshold: float = Field(gt=0)
    max_abs_threshold: float | None = Field(default=None, gt=0)
    # Per-step overrides of `threshold` (leaf index -> threshold), where early and late steps differ.
    step_thresholds: dict[int, float] = Field(default_factory=dict)
    image_digest: str | None = None
    runtime: str | None = None
    calibrated_at: str | None = None
    notes: str | None = None

    def key(self) -> tuple[str, str, str]:
        return self.profile_id, self.hardware_class, self.executor_class

    def threshold_for(self, step: int) -> float:
        return self.step_thresholds.get(step, self.threshold)

    def accepts(self, distance: StepDistance, step: int) -> bool:
        if not distance.finite:
            return False
        if distance.rel_l2 > self.threshold_for(step):
            return False
        return self.max_abs_threshold is None or distance.max_abs_rel <= self.max_abs_threshold


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    entries: list[CalibrationEntry] = Field(default_factory=list)

    def lookup(self, profile_id: str, hardware_class: str, executor_class: str | None) -> CalibrationEntry | None:
        """The entry for this comparison; an exact executor class wins over the wildcard."""
        candidates = [e for e in self.entries if e.profile_id == profile_id and e.hardware_class == hardware_class]
        exact = [e for e in candidates if executor_class is not None and e.executor_class == executor_class]
        wildcard = [e for e in candidates if e.executor_class == ANY_EXECUTOR]
        return (exact or wildcard or [None])[0]

    def with_entry(self, entry: CalibrationEntry) -> Calibration:
        """A copy with `entry` added, replacing any entry for the same comparison."""
        kept = [e for e in self.entries if e.key() != entry.key()]
        return Calibration(entries=sorted([*kept, entry], key=lambda e: e.key()))


def load_calibration(path: str | Path | None = None) -> Calibration:
    """The calibration at `path`, or the one shipped with kuno-protocol (empty until GPU runs fill it)."""
    if path is None:
        text = resources.files(__package__).joinpath(CALIBRATION_FILE).read_text()
    else:
        text = Path(path).read_text()
    return Calibration.model_validate(json.loads(text))


def summarize(values: Iterable[float]) -> DistanceStats:
    np = _np()
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    if array.size == 0:
        raise CalibrationError("no samples")
    if not np.all(np.isfinite(array)):
        raise CalibrationError("samples contain non-finite distances")
    quantile = lambda q: float(np.quantile(array, q, method="higher"))  # noqa: E731 — never below an observed sample
    return DistanceStats(
        samples=int(array.size), mean=float(array.mean()), p50=quantile(0.5), p99=quantile(0.99), p999=quantile(0.999), max=float(array.max())
    )


def propose_threshold(
    honest: Sequence[float], substituted: Sequence[float] | None = None, *, margin: float = 2.0, min_samples: int = 200
) -> float:
    """A threshold from measured distances.

    `margin` × the largest honest distance, so no honest sample in the calibration set would
    have failed. With substituted samples, the threshold must also sit below the smallest of
    them; if honest × margin reaches into them it falls back to the geometric midpoint, and if
    the two distributions overlap there is no safe threshold at all.
    """
    if len(honest) < min_samples:
        raise CalibrationError(f"{len(honest)} honest samples; at least {min_samples} are needed")
    if margin < 1.0:
        raise CalibrationError("the margin must be at least 1")
    worst_honest = summarize(honest).max
    threshold = max(worst_honest * margin, FLOOR)
    if substituted:
        best_cheat = min(float(v) for v in substituted)
        if best_cheat <= worst_honest:
            raise CalibrationError(
                f"honest ({worst_honest:.3g}) and substituted ({best_cheat:.3g}) distances overlap: this metric cannot separate them here"
            )
        if threshold >= best_cheat:
            threshold = math.sqrt(max(worst_honest, FLOOR) * best_cheat)
    return threshold


def comparison_for(hardware_class) -> str:
    """"bitwise" or "tolerance" for a profiles.HardwareClass (older data without the field is bitwise)."""
    return getattr(hardware_class, "comparison", BITWISE) or BITWISE
