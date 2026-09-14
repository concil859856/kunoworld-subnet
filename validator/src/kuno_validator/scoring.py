"""Serving-mechanism scoring.

score_m = Σ_family split_f × [(1 − s_f) × VCU_m,f / Σ_miners VCU_f + s_f × C_m,f / Σ_miners C_f]
                                                                        for miners that pass the gates
VCU_m,f = Σ_jobs rate_tier × profile weight × requested seconds

  VCU        verified video compute units: the profile's per-second weight × the seconds
             the customer requested in the job's public parameters, for each receipt that
             survived `ledger.audit_ledger` (customer and canary jobs alike). The miner's
             own reported duration is only checked, never paid.
  rate_t     the tier rate of the enclave that ran the job (open_tier.py): 1 for the confidential tier,
             KUNO_OPEN_TIER_RATE (default 0.5) for the open tier; entries without a tier earn at 1
  split_f    the owner-signed switch's emission share for each family in use
  gates      a currently attested enclave; success rate ≥ min_success once a miner has
             at least min_samples finished jobs in the window; and no penalty in the
             window (a failed canary or a cross-miner replay zeroes the miner, and so do
             `hardware_conflicts` and an unmet collateral requirement, see collateral.py)

Capacity pay (VALIDATING.md, "Capacity pay") only when the switch's capacity_share s is above 0. At 0, s_f = 0
everywhere and every score is exactly the VCU score:

  C_m,f      the miner's verified GPU-seconds in family f over the window (capacity.py: ready confidential-tier
             GPUs, checked by this validator's own challenges, in runs of at least capacity_min_uptime_s) × scale_f.
             Only for a miner that passes the gates and has a succeeded, credited confidential-tier job of the
             family in the window, and only in a family the switch enables and gives a target.
  avg_f      Σ_miners gated GPU-seconds_f / window_s: the window's average verified GPUs
  scale_f    min(1, target_f / avg_f): capacity beyond the target dilutes instead of adding emission
  s_f        s × min(1, avg_f / target_f), so an undersubscribed target doesn't overpay the few miners present.
             A family with capacity credit but no VCU gives the job part to capacity (s_f = 1); a family
             without capacity credit keeps it all for VCU (s_f = 0).

Only failures the miner is responsible for count against it: crashes, timeouts,
or going offline with work assigned. Customer errors (safety blocks, bad inputs)
don't.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from kuno_protocol.profiles import ModelProfile
from kuno_protocol.switch import SwitchConfig
from kuno_protocol.tiers import OPEN

MINER_FAULT_CODES = frozenset({"internal_error", "timeout", "enclave_unavailable", "queue_timeout"})


@dataclass
class MinerScore:
    hotkey: str
    work: dict[str, float] = field(default_factory=dict)
    succeeded: int = 0
    failed: int = 0
    attested: bool = False
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0
    # Observations that did not disqualify the miner (e.g. an uncredited duration mismatch).
    flags: list[str] = field(default_factory=list)
    # Families with a succeeded, credited confidential-tier job in the window: capacity pay needs one per family.
    served: set[str] = field(default_factory=set)
    # Capacity pay: gated GPU-seconds per family after the target cap. Empty unless the switch pays for capacity.
    capacity: dict[str, float] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.succeeded + self.failed
        return self.succeeded / total if total else 1.0


@dataclass
class FamilyCapacity:
    """One family's capacity over the window: gated verified GPU-seconds against the switch's target."""

    target: int
    window_s: float
    # Σ gated GPU-seconds, before the cap.
    gpu_seconds: float = 0.0
    # s_f: the part of the family's split paid for capacity in VCU mode (set by compute_scores).
    blend: float = 0.0

    @property
    def average_gpus(self) -> float:
        return self.gpu_seconds / self.window_s if self.window_s > 0 else 0.0

    @property
    def scale(self) -> float:
        """Every miner's credit multiplier: target / average when the window averaged more GPUs than the target."""
        return min(1.0, self.target / self.average_gpus) if self.average_gpus > 0 else 1.0

    @property
    def utilization(self) -> float:
        return min(1.0, self.average_gpus / self.target) if self.target > 0 else 0.0


@dataclass
class CapacityCredit:
    """The verified GPU-time going into a round, and what scoring made of it."""

    # hotkey -> family -> GPU-seconds in the window from runs that met the uptime rule (capacity.py), before any gate
    gpu_seconds: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    # family -> totals, filled in by compute_scores (empty when the switch doesn't pay for capacity)
    families: dict[str, FamilyCapacity] = field(default_factory=dict)


def billable_seconds(entry: dict) -> float | None:
    """Seconds to pay for: the audited public duration, or None when the entry must not earn."""
    if entry.get("credit") is False:
        return None
    seconds = entry.get("billable_s", entry.get("duration_s"))
    return float(seconds) if isinstance(seconds, (int, float)) else None


def compute_scores(
    ledger: list[dict],
    attested_hotkeys: set[str],
    profiles: dict[str, ModelProfile],
    switch: SwitchConfig,
    now: float,
    window_s: float = 86400.0,
    min_success: float = 0.98,
    min_samples: int = 20,
    penalties: Mapping[str, list[str]] | None = None,
    flags: Mapping[str, list[str]] | None = None,
    tier_rates: Mapping[str, float] | None = None,
    capacity: CapacityCredit | None = None,
) -> dict[str, MinerScore]:
    """Scores an already-audited ledger. Pass raw gateway rows through `audit_ledger` first. `capacity` is the window's
    verified GPU-time (capacity.py); its per-family totals are filled in here."""
    miners: dict[str, MinerScore] = {}
    for entry in ledger:
        hotkey = entry.get("miner_hotkey")
        if not hotkey or (entry.get("finished_at") or 0) < now - window_s:
            continue
        miner = miners.setdefault(hotkey, MinerScore(hotkey))
        profile = profiles.get(entry["profile_id"])
        if entry["status"] == "succeeded" and entry.get("receipt") and profile is not None:
            seconds = billable_seconds(entry)
            if seconds is not None:
                rate = float((tier_rates or {}).get(entry.get("tier") or "", 1.0))
                miner.work[profile.family] = miner.work.get(profile.family, 0.0) + profile.vcu(seconds) * rate
                if entry.get("tier") != OPEN:
                    miner.served.add(profile.family)
            miner.succeeded += 1
        elif entry["status"] == "failed" and entry.get("error_code") in MINER_FAULT_CODES:
            miner.failed += 1

    for hotkey in attested_hotkeys:
        miners.setdefault(hotkey, MinerScore(hotkey))
    for hotkey in penalties or {}:
        miners.setdefault(hotkey, MinerScore(hotkey))
    for miner in miners.values():
        miner.attested = miner.hotkey in attested_hotkeys
        if not miner.attested:
            miner.reasons.append("no currently attested enclave")
        if miner.succeeded + miner.failed >= min_samples and miner.success_rate < min_success:
            miner.reasons.append(f"success rate {miner.success_rate:.1%} is below {min_success:.0%}")
        miner.reasons.extend((penalties or {}).get(miner.hotkey, []))
        miner.flags.extend((flags or {}).get(miner.hotkey, []))

    eligible = [m for m in miners.values() if not m.reasons]
    # With capacity_share 0 nothing below touches capacity, so every score is exactly the VCU score.
    capacity_families: dict[str, FamilyCapacity] = {}
    if capacity is not None:
        if switch.capacity_share > 0:
            capacity_families = gate_capacity(miners, capacity.gpu_seconds, switch, window_s)
        capacity.families = capacity_families
    families = {f for m in eligible for f, w in m.work.items() if w > 0 and switch.family_enabled(f)}
    families |= set(capacity_families)
    split = {f: max(switch.emission_split.get(f, 0.0), 0.0) for f in families}
    total_split = sum(split.values())
    if total_split == 0:
        split = {f: 1.0 for f in families}
        total_split = float(len(families))
    for family in families:
        family_total = sum(m.work.get(family, 0.0) for m in eligible)
        capacity_total = sum(m.capacity.get(family, 0.0) for m in eligible)
        if capacity_total <= 0:
            for miner in eligible:
                miner.score += split[family] / total_split * miner.work.get(family, 0.0) / family_total
            continue
        ready = capacity_families[family]
        # Nobody did verified work in the family: the job part's share goes to capacity.
        ready.blend = switch.capacity_share * ready.utilization if family_total > 0 else 1.0
        for miner in eligible:
            job = miner.work.get(family, 0.0) / family_total if family_total > 0 else 0.0
            part = (1.0 - ready.blend) * job + ready.blend * miner.capacity.get(family, 0.0) / capacity_total
            miner.score += split[family] / total_split * part
    return miners


def gate_capacity(
    miners: Mapping[str, MinerScore], gpu_seconds: Mapping[str, Mapping[str, float]], switch: SwitchConfig, window_s: float
) -> dict[str, FamilyCapacity]:
    """Credits verified GPU-seconds to miners that pass every gate, scales each family down to the switch's target,
    and flags what was credited or withheld. Sets `MinerScore.capacity`; returns the per-family totals."""
    families: dict[str, FamilyCapacity] = {}
    for hotkey, by_family in sorted(gpu_seconds.items()):
        miner = miners.get(hotkey)
        if miner is None or miner.reasons:
            continue  # zeroed, and its reasons already say why
        for family, seconds in sorted(by_family.items()):
            if seconds <= 0:
                continue
            target = switch.capacity_target(family)
            if not switch.family_enabled(family):
                why = "the family is switched off"
            elif target <= 0:
                why = "the switch sets no capacity target for it"
            elif family not in miner.served:
                why = "no succeeded confidential-tier job of the family in the window"
            else:
                miner.capacity[family] = seconds
                families.setdefault(family, FamilyCapacity(target, window_s)).gpu_seconds += seconds
                continue
            miner.flags.append(f"capacity {family}: {seconds / 3600:.2f} verified GPU-hours earn nothing ({why})")
    for family, total in sorted(families.items()):
        scale = total.scale
        for miner in miners.values():
            if family not in miner.capacity:
                continue
            miner.capacity[family] *= scale
            capped = (
                f", scaled by {scale:.3f} (the window averaged {total.average_gpus:.2f} GPUs for a target of {total.target})"
                if scale < 1 else ""
            )
            miner.flags.append(f"capacity {family}: {miner.capacity[family] / 3600:.2f} GPU-hours credited{capped}")
    return families


def hardware_conflicts(sightings: Mapping[str, Mapping], now: float, window_s: float) -> dict[str, list[str]]:
    """One machine, one miner: zero-weight reasons for hotkeys that shared verified hardware.

    `sightings` maps a hardware token to {"kind": ..., "hotkeys": {hotkey: [first_seen, last_seen]}},
    recorded only from the validator's own successful challenge verdicts. Among the hotkeys
    that showed a token inside the window, the one that showed it strictly first keeps it
    and every later hotkey is zeroed. If several hotkeys showed it first in the same round,
    all of them are zeroed: one device can't be in two VMs at once, so a concurrent sighting
    means a relay or a split host serving several hotkeys, and there is no honest first one.
    """
    lost: dict[str, dict[str, set[str]]] = {}
    for token, sighting in sorted(sightings.items()):
        seen = {
            hotkey: (float(first), float(last))
            for hotkey, (first, last) in (sighting.get("hotkeys") or {}).items()
            if float(last) >= now - window_s
        }
        if len(seen) < 2:
            continue
        earliest = min(first for first, _ in seen.values())
        keepers = sorted(hotkey for hotkey, (first, _) in seen.items() if first == earliest)
        kind = str(sighting.get("kind", "hardware"))
        for hotkey in sorted(seen):
            if keepers == [hotkey]:
                continue
            if hotkey in keepers:
                why = f"also attested by {', '.join(k for k in keepers if k != hotkey)} in the same round"
            else:
                why = f"first attested by {', '.join(keepers)}"
            lost.setdefault(hotkey, {}).setdefault(why, set()).add(f"{kind}:{token}")
    penalties: dict[str, list[str]] = {}
    for hotkey, groups in lost.items():
        for why, items in sorted(groups.items()):
            kinds = ", ".join(sorted({item.split(":", 1)[0].replace("_", " ") for item in items}))
            noun = "identity" if len(items) == 1 else "identities"
            penalties.setdefault(hotkey, []).append(f"shares {len(items)} verified hardware {noun} ({kinds}) {why}")
    return penalties


def normalize(scores: dict[str, MinerScore]) -> dict[str, float]:
    total = sum(m.score for m in scores.values())
    if total <= 0:
        return {}
    return {hotkey: m.score / total for hotkey, m in scores.items() if m.score > 0}
