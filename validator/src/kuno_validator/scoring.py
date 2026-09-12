"""Serving-mechanism scoring.

score_m = Σ_family split_f × (VCU_m,f / Σ_miners VCU_f)     for miners that pass the gates

  VCU        verified video compute units: the profile's per-second weight × seconds
             of video in each enclave-signed receipt (customer and canary jobs alike)
  split_f    the owner-signed switch's emission share for each family in use
  gates      a currently attested enclave, and success rate ≥ min_success once a
             miner has at least min_samples finished jobs in the window

Only failures the miner is responsible for count against it: crashes, timeouts,
or going offline with work assigned. Customer errors (safety blocks, bad inputs)
don't.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kuno_protocol.profiles import ModelProfile
from kuno_protocol.switch import SwitchConfig

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

    @property
    def success_rate(self) -> float:
        total = self.succeeded + self.failed
        return self.succeeded / total if total else 1.0


def compute_scores(
    ledger: list[dict],
    attested_hotkeys: set[str],
    profiles: dict[str, ModelProfile],
    switch: SwitchConfig,
    now: float,
    window_s: float = 86400.0,
    min_success: float = 0.98,
    min_samples: int = 20,
) -> dict[str, MinerScore]:
    miners: dict[str, MinerScore] = {}
    for entry in ledger:
        hotkey = entry.get("miner_hotkey")
        if not hotkey or (entry.get("finished_at") or 0) < now - window_s:
            continue
        miner = miners.setdefault(hotkey, MinerScore(hotkey))
        profile = profiles.get(entry["profile_id"])
        if entry["status"] == "succeeded" and entry.get("receipt") and profile is not None:
            seconds = float(entry["receipt"]["body"]["video"]["duration_s"])
            miner.work[profile.family] = miner.work.get(profile.family, 0.0) + profile.vcu(seconds)
            miner.succeeded += 1
        elif entry["status"] == "failed" and entry.get("error_code") in MINER_FAULT_CODES:
            miner.failed += 1

    for hotkey in attested_hotkeys:
        miners.setdefault(hotkey, MinerScore(hotkey))
    for miner in miners.values():
        miner.attested = miner.hotkey in attested_hotkeys
        if not miner.attested:
            miner.reasons.append("no currently attested enclave")
        if miner.succeeded + miner.failed >= min_samples and miner.success_rate < min_success:
            miner.reasons.append(f"success rate {miner.success_rate:.1%} is below {min_success:.0%}")

    eligible = [m for m in miners.values() if not m.reasons]
    families = {f for m in eligible for f, w in m.work.items() if w > 0 and switch.family_enabled(f)}
    split = {f: max(switch.emission_split.get(f, 0.0), 0.0) for f in families}
    total_split = sum(split.values())
    if total_split == 0:
        split = {f: 1.0 for f in families}
        total_split = float(len(families))
    for family in families:
        family_total = sum(m.work.get(family, 0.0) for m in eligible)
        for miner in eligible:
            miner.score += split[family] / total_split * miner.work.get(family, 0.0) / family_total
    return miners


def normalize(scores: dict[str, MinerScore]) -> dict[str, float]:
    total = sum(m.score for m in scores.values())
    if total <= 0:
        return {}
    return {hotkey: m.score / total for hotkey, m in scores.items() if m.score > 0}
