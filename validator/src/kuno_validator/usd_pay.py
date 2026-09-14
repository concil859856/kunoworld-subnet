"""USD-denominated miner pay for the serving mechanism (`KUNO_PAY_MODE=usd`).

The default (`KUNO_PAY_MODE=vcu`) is today's scoring: the miner emission is split in proportion to
verified video compute units (scoring.py). USD mode keeps every gate and penalty that scoring
applies (attestation, reliability, canaries, replays, step audits, hardware dedupe, collateral,
open-tier admission and fraud) and changes only what a gated miner's verified work is worth:

    usd_owed_i  = Σ billable seconds × rate(profile, tier)   over miner i's credited jobs in the scoring
                                                             window, rates from the owner-signed rate card
    owed_i      = usd_owed_i × tempo_seconds / window_s      the window's average per tempo
    pool_usd    = serving miners' alpha per tempo × TAO per alpha × USD per TAO
    weight_i    = owed_i / pool_usd, then renormalized

  Capacity pay (VALIDATING.md, "Capacity pay"), when the switch's capacity_share is above 0, adds to usd_owed_i:
    capacity_i  = Σ_family credited GPU-hours × gpu_hour_usd(family)    scoring.py's gated GPU-hours, already scaled
                                                                        down to the switch's targets
    Σ capacity_i × tempo_seconds / window_s is limited to capacity_share × pool_usd: above it, every miner's capacity
    owed is scaled down by the same factor before it is added to its job owed.

  * Undersubscribed (Σ weight < 1): the residual is handled by `KUNO_PAY_RESIDUAL`. The only accepted
    value is `renormalize` (the default): weights are scaled up to sum to 1, so miners receive the whole
    pool and nothing is burned. `recycle` (the residual to the owner uid) is documented in VALIDATING.md
    and refused here: since June 2026 the withheld share (`MinerBurned`) scales the subnet's TAO
    emission share down by (1 − MinerBurned), and recycling doesn't avoid that.
  * Oversubscribed (Σ weight > 1): weights are renormalized down; every miner gets the same fraction
    of what it is owed.

Billable seconds and tiers come from the audited ledger exactly as scoring sees them; the card's
per-tier rate replaces the VCU weight, `KUNO_OPEN_TIER_RATE` and the switch's family split (a family
the switch turns off still earns nothing).

Failing closed: no accepted rate card, an unreadable chain, or a TAO/USD rate without two fresh
sources that agree raises `PayUnavailable`, and the caller keeps the previous weights.

Each priced round is logged and appended as one JSON line to the pay report, with the subsidy ratio
(emission value ÷ miner USD owed, per tempo) and the emission-to-revenue KPI (all participants'
emission value ÷ customer revenue at list price, over the window).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import ValidationError

from kuno_protocol.profiles import ModelProfile
from kuno_protocol.rate_card import RateCard, SignedRateCard
from kuno_protocol.schemas import GenerationParams
from kuno_protocol.switch import SwitchConfig
from kuno_protocol.tiers import CONFIDENTIAL

from .chain import SERVING_MECHID
from .collateral import NETWORK_ENDPOINTS
from .emission import EmissionReader, SubnetEmission, SubstrateEmissionReader
from .price_feeds import DEFAULT_MAX_AGE_S, DEFAULT_TOLERANCE, PriceUnavailable, TaoUsd, TaoUsdOracle
from .scoring import CapacityCredit, FamilyCapacity, MinerScore, billable_seconds

log = logging.getLogger("kuno.validator.usd_pay")

VCU, USD = "vcu", "usd"
PAY_MODES = (VCU, USD)
RENORMALIZE, RECYCLE = "renormalize", "recycle"
RESIDUALS = (RENORMALIZE,)
RECYCLE_REFUSED = (
    "KUNO_PAY_RESIDUAL=recycle is documented but not implemented: weight sent to the owner uid is withheld miner "
    "emission, and since June 2026 the withheld share (MinerBurned) scales the subnet's TAO emission share down by "
    "(1 - MinerBurned), whether it is burned or recycled. Use renormalize (see VALIDATING.md, 'USD-denominated pay')."
)


class PayUnavailable(RuntimeError):
    """This round can't be priced; keep the previous weights."""


@dataclass
class PayPolicy:
    mode: str = VCU
    residual: str = RENORMALIZE
    rate_card_path: Path | None = None
    price_tolerance: float = DEFAULT_TOLERANCE
    price_max_age_s: float = DEFAULT_MAX_AGE_S

    def __post_init__(self) -> None:
        if self.mode not in PAY_MODES:
            raise ValueError(f"KUNO_PAY_MODE must be one of {', '.join(PAY_MODES)}, not {self.mode!r}")
        if self.residual == RECYCLE:
            raise ValueError(RECYCLE_REFUSED)
        if self.residual not in RESIDUALS:
            raise ValueError(f"KUNO_PAY_RESIDUAL must be {RENORMALIZE}, not {self.residual!r}")

    @property
    def usd(self) -> bool:
        return self.mode == USD

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> PayPolicy:
        """KUNO_PAY_MODE (vcu|usd, default vcu), KUNO_PAY_RESIDUAL (renormalize), KUNO_RATE_CARD (signed card path),
        KUNO_PAY_PRICE_TOLERANCE (default 0.02), KUNO_PAY_PRICE_MAX_AGE_S (default 900)."""
        card = (env.get("KUNO_RATE_CARD") or "").strip()
        tolerance = (env.get("KUNO_PAY_PRICE_TOLERANCE") or "").strip()
        max_age = (env.get("KUNO_PAY_PRICE_MAX_AGE_S") or "").strip()
        return cls(
            mode=(env.get("KUNO_PAY_MODE") or VCU).strip().lower(),
            residual=(env.get("KUNO_PAY_RESIDUAL") or RENORMALIZE).strip().lower(),
            rate_card_path=Path(card) if card else None,
            price_tolerance=float(tolerance) if tolerance else DEFAULT_TOLERANCE,
            price_max_age_s=float(max_age) if max_age else DEFAULT_MAX_AGE_S,
        )


@dataclass
class OwedWork:
    """What gated miners are owed over the window, and what customers paid for the window's jobs."""

    owed_usd: dict[str, float] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    # "profile@tier" -> verified seconds the card has no rate for (they earn nothing)
    unpriced: dict[str, float] = field(default_factory=dict)
    revenue_usd: float = 0.0
    revenue_jobs: int = 0
    revenue_unknown_jobs: int = 0
    # Capacity pay, before the capacity_share cap: hotkey -> USD for its credited GPU-hours at the card's family rates,
    # hotkey -> and family -> priced GPU-hours, family -> credited GPU-hours the card has no rate for (they earn nothing),
    # and scoring's per-family figures (target, average GPUs, target scale, utilization).
    capacity_usd: dict[str, float] = field(default_factory=dict)
    capacity_gpu_hours: dict[str, float] = field(default_factory=dict)
    capacity_family_gpu_hours: dict[str, float] = field(default_factory=dict)
    capacity_unpriced: dict[str, float] = field(default_factory=dict)
    capacity_families: dict[str, dict[str, float]] = field(default_factory=dict)


def list_price_usd(entry: Mapping, profile: ModelProfile) -> float | None:
    """What the customer paid for a job: the ledger's `price_usd` when the gateway publishes one, else the
    profile's public list price for the job's params. Discounts, credits and refunds are not visible here."""
    value = entry.get("price_usd")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    if entry.get("params") is not None:
        try:
            return profile.price_usd(GenerationParams.model_validate(entry["params"]))
        except (ValidationError, ValueError):
            return None
    rate = profile.pricing.usd_per_second.get(entry.get("resolution") or "")
    seconds = entry.get("billable_s", entry.get("duration_s"))
    if rate is None or not isinstance(seconds, (int, float)):
        return None
    return round(rate * float(seconds), 4)


def usd_owed(
    entries: Iterable[dict],
    eligible: set[str],
    card: RateCard,
    profiles: Mapping[str, ModelProfile],
    switch: SwitchConfig,
    now: float,
    window_s: float,
) -> OwedWork:
    """Prices an audited ledger (after `audit_ledger` and `apply_tiers`) for the miners that passed every gate."""
    work = OwedWork()
    for entry in entries:
        if (entry.get("finished_at") or 0) < now - window_s:
            continue
        if entry.get("status") != "succeeded" or not entry.get("receipt"):
            continue
        profile = profiles.get(entry.get("profile_id") or "")
        if profile is None:
            continue
        if not entry.get("replay_of"):
            price = list_price_usd(entry, profile)
            if price is None:
                work.revenue_unknown_jobs += 1
            else:
                work.revenue_usd += price
                work.revenue_jobs += 1
        hotkey = entry.get("miner_hotkey")
        if hotkey not in eligible or not switch.family_enabled(profile.family):
            continue
        seconds = billable_seconds(entry)
        if seconds is None:
            continue
        tier = entry.get("tier") or CONFIDENTIAL
        rate = card.rate(profile.id, tier)
        if rate is None:
            key = f"{profile.id}@{tier}"
            work.unpriced[key] = work.unpriced.get(key, 0.0) + seconds
            continue
        work.owed_usd[hotkey] = work.owed_usd.get(hotkey, 0.0) + seconds * rate
        work.seconds[hotkey] = work.seconds.get(hotkey, 0.0) + seconds
    return work


def capacity_owed(
    work: OwedWork, scores: Mapping[str, MinerScore], card: RateCard, families: Mapping[str, FamilyCapacity] | None = None
) -> OwedWork:
    """Adds capacity pay to `work`: each gated miner's credited GPU-hours (`MinerScore.capacity`, where scoring already
    applied every gate and the target cap) × the card's `gpu_hour_usd` for the family. Empty unless the switch pays."""
    for hotkey, miner in sorted(scores.items()):
        if miner.reasons:
            continue
        for family, seconds in sorted(miner.capacity.items()):
            hours = seconds / 3600.0
            if hours <= 0:
                continue
            rate = card.gpu_hour_rate(family)
            if rate is None:
                work.capacity_unpriced[family] = work.capacity_unpriced.get(family, 0.0) + hours
                continue
            work.capacity_usd[hotkey] = work.capacity_usd.get(hotkey, 0.0) + hours * rate
            work.capacity_gpu_hours[hotkey] = work.capacity_gpu_hours.get(hotkey, 0.0) + hours
            work.capacity_family_gpu_hours[family] = work.capacity_family_gpu_hours.get(family, 0.0) + hours
    for family, total in sorted((families or {}).items()):
        work.capacity_families[family] = {
            "target": float(total.target), "average_gpus": total.average_gpus, "target_scale": total.scale,
            "utilization": total.utilization,
        }
    return work


@dataclass
class PayReport:
    at: float
    netuid: int
    mechid: int
    block: str | None
    residual: str
    regime: str  # undersubscribed | oversubscribed | balanced | no_work
    window_s: float
    tempo_seconds: float
    rate_card_issued_at: int
    rate_card_placeholder: bool
    usd_per_tao: float
    tao_per_alpha: float
    usd_per_alpha: float
    price_sources: dict[str, float]
    moving_tao_per_alpha: float | None
    miner_burned: float | None
    miner_alpha_per_tempo: float
    pool_usd_per_tempo: float
    # Job and capacity owed together (capacity after its cap).
    owed_usd_window: float
    owed_usd_per_tempo: float
    # Capacity pay: the switch's capacity_share; what credited GPU-hours are owed per tempo at card rates, the limit of
    # capacity_share × pool_usd_per_tempo, the multiplier that keeps it under the limit (1 when already under), what is
    # owed after it per tempo and over the window; priced GPU-hours per family, GPU-hours without a card rate, and
    # scoring's per-family figures (target, average GPUs, target scale, utilization).
    capacity_share: float
    capacity_uncapped_usd_per_tempo: float
    capacity_limit_usd_per_tempo: float
    capacity_scale: float
    capacity_usd_per_tempo: float
    capacity_usd_window: float
    capacity_gpu_hours: dict[str, float]
    capacity_unpriced: dict[str, float]
    capacity_families: dict[str, dict[str, float]]
    # Σ owed_i / pool_usd before renormalizing: below 1 undersubscribed, above 1 oversubscribed.
    subscription: float
    # Emission value ÷ miner USD owed (per tempo). Above 1: emissions subsidize miners beyond what they're owed.
    subsidy_ratio: float | None
    emission_usd_window: float
    revenue_usd_window: float
    revenue_jobs: int
    revenue_unknown_jobs: int
    # All participants' emission value ÷ customer revenue over the window.
    emission_to_revenue: float | None
    miners: dict[str, dict[str, float]]
    unpriced: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def settle(
    work: OwedWork, emission: SubnetEmission, tao: TaoUsd, card: RateCard, now: float, window_s: float,
    residual: str = RENORMALIZE, capacity_share: float = 0.0,
) -> tuple[dict[str, float], PayReport]:
    """Turns USD owed into weights against the pool's USD value. Pure: no chain, no network. `capacity_share` is the
    switch's: capacity owed per tempo is held to that share of the pool."""
    if residual not in RESIDUALS:
        raise ValueError(RECYCLE_REFUSED if residual == RECYCLE else f"unknown residual policy {residual!r}")
    if window_s <= 0:
        raise ValueError("the scoring window must be positive")
    usd_per_alpha = emission.tao_per_alpha * tao.usd_per_tao
    pool = emission.miner_alpha_per_tempo * usd_per_alpha
    if not (pool > 0 and math.isfinite(pool)):
        raise PayUnavailable(
            f"the serving miners' emission is worth ${pool:.4f} per tempo "
            f"({emission.miner_alpha_per_tempo:.4f} alpha at {emission.tao_per_alpha:.6f} TAO); nothing to price against"
        )
    scale = emission.tempo_seconds / window_s
    owed_tempo = {hotkey: usd * scale for hotkey, usd in work.owed_usd.items() if usd > 0}
    # Capacity pay: above capacity_share × pool, every miner's capacity owed is scaled down by the same factor.
    capacity_limit = max(capacity_share, 0.0) * pool
    capacity_uncapped = {hotkey: usd * scale for hotkey, usd in work.capacity_usd.items() if usd > 0}
    uncapped_total = sum(capacity_uncapped.values())
    capacity_scale = min(1.0, capacity_limit / uncapped_total) if uncapped_total > 0 else 1.0
    capacity_tempo = {hotkey: value * capacity_scale for hotkey, value in capacity_uncapped.items()}
    for hotkey, value in capacity_tempo.items():
        if value > 0:
            owed_tempo[hotkey] = owed_tempo.get(hotkey, 0.0) + value
    raw = {hotkey: value / pool for hotkey, value in owed_tempo.items()}
    subscription = sum(raw.values())
    if subscription <= 0:
        regime, weights = "no_work", {}
    else:
        regime = "undersubscribed" if subscription < 1 else "oversubscribed" if subscription > 1 else "balanced"
        # renormalize: scaled up when undersubscribed (no burn), scaled down when oversubscribed.
        weights = {hotkey: value / subscription for hotkey, value in raw.items()}
    owed_per_tempo = sum(owed_tempo.values())
    capacity_window = {hotkey: usd * capacity_scale for hotkey, usd in work.capacity_usd.items() if usd > 0}
    emission_usd = emission.emission_alpha(window_s) * usd_per_alpha
    report = PayReport(
        at=now, netuid=emission.netuid, mechid=SERVING_MECHID, block=emission.block, residual=residual, regime=regime,
        window_s=window_s, tempo_seconds=emission.tempo_seconds,
        rate_card_issued_at=card.issued_at, rate_card_placeholder=card.placeholder,
        usd_per_tao=tao.usd_per_tao, tao_per_alpha=emission.tao_per_alpha, usd_per_alpha=usd_per_alpha,
        price_sources=dict(tao.quotes), moving_tao_per_alpha=emission.moving_tao_per_alpha, miner_burned=emission.miner_burned,
        miner_alpha_per_tempo=emission.miner_alpha_per_tempo, pool_usd_per_tempo=pool,
        owed_usd_window=sum(work.owed_usd.values()) + sum(capacity_window.values()), owed_usd_per_tempo=owed_per_tempo,
        capacity_share=capacity_share, capacity_uncapped_usd_per_tempo=uncapped_total, capacity_limit_usd_per_tempo=capacity_limit,
        capacity_scale=capacity_scale, capacity_usd_per_tempo=sum(capacity_tempo.values()), capacity_usd_window=sum(capacity_window.values()),
        capacity_gpu_hours=dict(work.capacity_family_gpu_hours), capacity_unpriced=dict(work.capacity_unpriced),
        capacity_families={family: dict(figures) for family, figures in work.capacity_families.items()},
        subscription=subscription,
        subsidy_ratio=pool / owed_per_tempo if owed_per_tempo > 0 else None,
        emission_usd_window=emission_usd, revenue_usd_window=work.revenue_usd, revenue_jobs=work.revenue_jobs,
        revenue_unknown_jobs=work.revenue_unknown_jobs,
        emission_to_revenue=emission_usd / work.revenue_usd if work.revenue_usd > 0 else None,
        miners={
            hotkey: {
                "usd_owed": work.owed_usd.get(hotkey, 0.0) + capacity_window.get(hotkey, 0.0), "usd_owed_per_tempo": owed_tempo[hotkey],
                "seconds": work.seconds.get(hotkey, 0.0), "job_usd_owed": work.owed_usd.get(hotkey, 0.0),
                "capacity_usd_owed": capacity_window.get(hotkey, 0.0), "capacity_gpu_hours": work.capacity_gpu_hours.get(hotkey, 0.0),
                "raw_weight": raw[hotkey], "weight": weights.get(hotkey, 0.0),
            }
            for hotkey in sorted(owed_tempo)
        },
        unpriced=dict(work.unpriced),
    )
    return weights, report


class UsdPay:
    """The rate card, the chain and the price feeds for one validator. Keeps the accepted card across rounds."""

    def __init__(
        self,
        policy: PayPolicy,
        netuid: int | None,
        reader: EmissionReader | None,
        oracle: TaoUsdOracle,
        owner_public_key: bytes | None,
        report_path: Path | None = None,
        mechid: int = SERVING_MECHID,
    ):
        self.policy, self.netuid, self.reader, self.oracle = policy, netuid, reader, oracle
        self.owner_public_key = owner_public_key
        self.report_path = report_path
        self.mechid = mechid
        self.accepted: SignedRateCard | None = None
        self.last_report: PayReport | None = None

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], netuid: int | None, network: str, owner_public_key: bytes | None, state_path: Path | None
    ) -> UsdPay | None:
        """None unless KUNO_PAY_MODE=usd. KUNO_PAY_REPORT (JSON lines; default next to the state file), KUNO_CHAIN_ENDPOINT."""
        policy = PayPolicy.from_env(env)
        if not policy.usd:
            return None
        reader = None
        if netuid is not None:
            reader = SubstrateEmissionReader(env.get("KUNO_CHAIN_ENDPOINT") or NETWORK_ENDPOINTS.get(network, network))
        oracle = TaoUsdOracle(tolerance=policy.price_tolerance, max_age_s=policy.price_max_age_s)
        report = (env.get("KUNO_PAY_REPORT") or "").strip()
        if report:
            report_path: Path | None = Path(report)
        else:
            report_path = state_path.with_name(f"{state_path.stem}-pay.jsonl") if state_path is not None else None
        return cls(policy, netuid, reader, oracle, owner_public_key, report_path)

    # ------------------------------------------------------------ rate card

    def rate_card(self) -> RateCard:
        """The card at KUNO_RATE_CARD, accepted like the switch: owner-signed, and `issued_at` never going backwards.
        A missing or bad file keeps the card accepted before; with none accepted, the round can't be priced."""
        path, signed, problem = self.policy.rate_card_path, None, None
        if path is None:
            problem = "KUNO_RATE_CARD is not set"
        else:
            try:
                signed = SignedRateCard.model_validate_json(path.read_text())
            except FileNotFoundError:
                problem = f"no rate card at {path}"
            except (OSError, ValueError) as exc:
                problem = f"rate card {path} is unreadable ({type(exc).__name__})"
        if signed is not None:
            if self.owner_public_key is None:
                log.error("using an UNVERIFIED rate card: no owner public key is configured")
            elif not signed.verify(self.owner_public_key):
                signed, problem = None, f"rate card {path} is not signed by the owner key"
        last = self.accepted
        if signed is not None and last is not None:
            if signed.card.issued_at < last.card.issued_at:
                log.warning("rate card issued at %d is older than the accepted %d; ignoring the rollback", signed.card.issued_at, last.card.issued_at)
                signed = None
            elif signed.card.issued_at == last.card.issued_at and signed.card != last.card:
                log.warning("a different rate card carries the accepted issued_at %d; keeping the accepted one", last.card.issued_at)
                signed = None
        if signed is not None:
            self.accepted = signed
        elif problem is not None:
            if self.accepted is None:
                raise PayUnavailable(f"{problem}, and no rate card has been accepted")
            log.warning("%s; keeping the rate card issued at %d", problem, self.accepted.card.issued_at)
        assert self.accepted is not None
        card = self.accepted.card
        if card.placeholder:
            log.error("rate card issued at %d is a PLACEHOLDER: the owner has not set these rates", card.issued_at)
        return card

    def dump(self) -> dict | None:
        return self.accepted.model_dump(mode="json") if self.accepted is not None else None

    def load(self, data: Mapping | None) -> None:
        """Restores the accepted card, so a restart can't accept an older one; only if it verifies under the owner key."""
        if not data:
            return
        try:
            stored = SignedRateCard.model_validate(data)
        except ValidationError:
            log.warning("stored rate card is malformed; discarded")
            return
        if self.owner_public_key is None or stored.verify(self.owner_public_key):
            self.accepted = stored
        else:
            log.warning("stored rate card is not signed by the configured owner key; discarded")

    # ------------------------------------------------------------ a round

    def weights(
        self,
        scores: Mapping[str, MinerScore],
        entries: Iterable[dict],
        profiles: Mapping[str, ModelProfile],
        switch: SwitchConfig,
        now: float,
        window_s: float,
        capacity: CapacityCredit | None = None,
    ) -> dict[str, float]:
        """Serving weights for a scored round, or PayUnavailable. `scores` supplies the gates: a miner with any
        reason earns nothing, exactly as in VCU scoring. It also carries each miner's credited capacity, which
        `capacity` (the round's CapacityCredit) describes per family."""
        card = self.rate_card()
        eligible = {hotkey for hotkey, miner in scores.items() if not miner.reasons}
        work = usd_owed(entries, eligible, card, profiles, switch, now, window_s)
        capacity_owed(work, scores, card, capacity.families if capacity is not None else None)
        if self.reader is None or self.netuid is None:
            raise PayUnavailable("no chain is configured to read the emission pool from (run with --netuid)")
        try:
            emission = self.reader.read(self.netuid, self.mechid)
        except Exception as exc:  # network, decoding, a missing library: all mean "unpriced this round"
            raise PayUnavailable(f"the emission pool could not be read: {exc}") from exc
        try:
            tao = self.oracle.quote()
        except PriceUnavailable as exc:
            raise PayUnavailable(str(exc)) from exc
        weights, report = settle(work, emission, tao, card, now, window_s, self.policy.residual, switch.capacity_share)
        self.last_report = report
        self._log(report)
        self._export(report)
        return weights

    def _log(self, report: PayReport) -> None:
        def ratio(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        log.info(
            "usd pay: %s (%.3f of the pool owed); pool $%.2f per tempo (%.4f alpha × $%.4f); owed $%.2f per tempo; "
            "subsidy ratio %s; emission/revenue %s ($%.2f emission, $%.2f revenue over %.0fs)%s",
            report.regime, report.subscription, report.pool_usd_per_tempo, report.miner_alpha_per_tempo, report.usd_per_alpha,
            report.owed_usd_per_tempo, ratio(report.subsidy_ratio), ratio(report.emission_to_revenue), report.emission_usd_window,
            report.revenue_usd_window, report.window_s, " [PLACEHOLDER RATE CARD]" if report.rate_card_placeholder else "",
        )
        if report.capacity_gpu_hours or report.capacity_unpriced:
            log.info(
                "usd pay capacity: %.2f GPU-hours credited, owed $%.2f per tempo against a limit of $%.2f (capacity_share %.2f "
                "of the pool), scale %.3f",
                sum(report.capacity_gpu_hours.values()), report.capacity_uncapped_usd_per_tempo, report.capacity_limit_usd_per_tempo,
                report.capacity_share, report.capacity_scale,
            )
        for hotkey, miner in report.miners.items():
            log.info(
                "miner %s: owed $%.4f over the window ($%.6f per tempo; $%.4f of it for %.2f GPU-hours of capacity), weight %.4f",
                hotkey, miner["usd_owed"], miner["usd_owed_per_tempo"], miner["capacity_usd_owed"], miner["capacity_gpu_hours"], miner["weight"],
            )
        for key, seconds in sorted(report.unpriced.items()):
            log.warning("the rate card has no rate for %s: %.1f verified seconds earn nothing", key, seconds)
        for family, hours in sorted(report.capacity_unpriced.items()):
            log.warning("the rate card has no gpu_hour_usd for %s: %.2f credited GPU-hours earn nothing", family, hours)

    def _export(self, report: PayReport) -> None:
        if self.report_path is None:
            return
        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            with self.report_path.open("a", encoding="utf-8") as handle:
                handle.write(report.to_json() + "\n")
        except OSError as exc:
            log.error("could not append the pay report to %s: %s", self.report_path, exc)
