"""How a validator weighs open-tier (no TEE) miners against confidential ones (PRIVACY_MODES.md).

An open-tier miner's hardware, image and memory are not attested, so three things carry the
weight attestation carries for the confidential tier:

  rate        its verified work earns `open_rate` (default 0.75) of a confidential miner's VCU:
              the confidential tier serves private jobs, costs more to run, and its results are
              attested as well as audited (`KUNO_OPEN_TIER_RATE`). At 0.5 only RTX 4090/5090 open
              miners broke even at 60% utilization (research/pricing/costs.md §8.2).
  admission   a new open-tier hotkey earns nothing until it has passed `admission_probes`
              (default 5) of this validator's canaries in a row; an attributable canary or audit
              failure during probation starts the count again (`KUNO_OPEN_TIER_PROBES`). It is
              the validator-side half of Engy/SN53's probe admission: the gateway-side half keeps
              customer traffic off unadmitted miners.
  fraud       a succeeded *private* job whose receipt came from an enclave this validator itself
              verified as open tier zeroes the miner for the window. Private jobs are only ever
              sealed to confidential enclaves, so such a receipt means the miner (or a relay it
              colludes with) took content it must never see. Only the validator's own challenge
              verdicts set a tier used here, so a gateway feed can't frame a confidential miner.

Collateral per GPU for the open tier is in collateral.py (`KUNO_MIN_COLLATERAL_PER_GPU_OPEN`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from kuno_protocol.tiers import CONFIDENTIAL, OPEN

log = logging.getLogger("kuno.validator.open_tier")

DEFAULT_OPEN_RATE = 0.75
DEFAULT_ADMISSION_PROBES = 5


@dataclass
class TierPolicy:
    open_rate: float = DEFAULT_OPEN_RATE
    admission_probes: int = DEFAULT_ADMISSION_PROBES

    def __post_init__(self) -> None:
        if not 0.0 <= self.open_rate <= 1.0:
            raise ValueError("KUNO_OPEN_TIER_RATE must be between 0 and 1: open-tier work never earns more than confidential work")
        if self.admission_probes < 0:
            raise ValueError("KUNO_OPEN_TIER_PROBES cannot be negative")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> TierPolicy:
        rate = (env.get("KUNO_OPEN_TIER_RATE") or "").strip()
        probes = (env.get("KUNO_OPEN_TIER_PROBES") or "").strip()
        return cls(
            open_rate=float(rate) if rate else DEFAULT_OPEN_RATE,
            admission_probes=int(probes) if probes else DEFAULT_ADMISSION_PROBES,
        )

    def rates(self) -> dict[str, float]:
        return {CONFIDENTIAL: 1.0, OPEN: self.open_rate}


@dataclass
class _Probation:
    passed: int = 0
    admitted_at: float | None = None
    last_probe_at: float | None = None


@dataclass
class AdmissionTracker:
    """Admission probes per open-tier hotkey. Confidential hotkeys never need admission here."""

    required: int = DEFAULT_ADMISSION_PROBES
    records: dict[str, _Probation] = field(default_factory=dict)

    def record(self, hotkey: str, ok: bool, attributable: bool, at: float) -> None:
        """One probe outcome for an open-tier hotkey. Unattributable failures (the gateway, the validator) count for nothing."""
        entry = self.records.setdefault(hotkey, _Probation())
        entry.last_probe_at = at
        if entry.admitted_at is not None:
            return  # admitted: later failures cost weight through the ordinary penalties, not re-admission
        if ok:
            entry.passed += 1
            if entry.passed >= self.required:
                entry.admitted_at = at
                log.info("open-tier miner %s admitted after %d probes", hotkey, entry.passed)
        elif attributable:
            entry.passed = 0

    def admitted(self, hotkey: str) -> bool:
        if self.required <= 0:
            return True
        entry = self.records.get(hotkey)
        return entry is not None and entry.admitted_at is not None

    def progress(self, hotkey: str) -> tuple[int, int]:
        entry = self.records.get(hotkey)
        return (entry.passed if entry else 0), self.required

    def dump(self) -> dict:
        return {hotkey: [e.passed, e.admitted_at, e.last_probe_at] for hotkey, e in self.records.items()}

    def load(self, data: Mapping) -> None:
        for hotkey, item in (data or {}).items():
            try:
                passed, admitted_at, last = item
                self.records[hotkey] = _Probation(int(passed), None if admitted_at is None else float(admitted_at), None if last is None else float(last))
            except (TypeError, ValueError):
                continue


def apply_tiers(entries: Iterable[dict], tiers: Mapping[str, str], admission: AdmissionTracker) -> dict[str, list[str]]:
    """Annotates audited ledger entries in place with `tier` and withholds credit from unadmitted open-tier miners.

    `tiers` maps enclave id -> tier: this validator's own verdicts first, then the gateway's enclave feed. An enclave
    known to neither earns as confidential, as every enclave did before the open tier existed; a short-lived
    confidential worker this validator never challenged must not lose half its pay. Returns flags per hotkey for
    unadmitted open-tier miners.
    """
    flags: dict[str, list[str]] = {}
    for entry in entries:
        tier = tiers.get(entry.get("enclave_id") or "", CONFIDENTIAL)
        entry["tier"] = tier
        hotkey = entry.get("miner_hotkey")
        if tier == OPEN and hotkey and not admission.admitted(hotkey) and entry.get("status") == "succeeded":
            entry["credit"] = False
            flags.setdefault(hotkey, [])
    for hotkey in flags:
        passed, required = admission.progress(hotkey)
        flags[hotkey].append(f"open-tier admission: {passed}/{required} probes passed; work does not earn yet")
    return flags


def fraud_penalties(entries: Iterable[dict], verified_tiers: Mapping[str, str]) -> dict[str, list[str]]:
    """Zero-weight reasons for private-mode receipts from enclaves this validator verified as open tier.

    Rows without a `privacy` field (gateways from before privacy modes) are never judged.
    """
    penalties: dict[str, list[str]] = {}
    for entry in entries:
        if entry.get("status") != "succeeded" or not entry.get("receipt") or entry.get("privacy") != "private":
            continue
        if verified_tiers.get(entry.get("enclave_id") or "") != OPEN or not entry.get("miner_hotkey"):
            continue
        penalties.setdefault(entry["miner_hotkey"], []).append(
            f"fraud: open-tier enclave {entry['enclave_id']} produced a receipt for private job {entry.get('job_id')}"
        )
    return penalties


def open_tier_gpus(enclave: Mapping, gpus_per_worker: Mapping[str, int]) -> int:
    """GPUs an open-tier enclave must post collateral for. Nothing is attested, so take the larger of what it
    reports and what its capacity needs: under-reporting GPUs to save collateral would cap its own jobs."""
    hardware = enclave.get("hardware") or {}
    try:
        reported = int(hardware.get("gpu_count") or 0)
    except (TypeError, ValueError):
        reported = 0
    profiles = [p for p in enclave.get("profiles") or [] if p in gpus_per_worker]
    per_job = max([int(gpus_per_worker[p]) for p in profiles] or [1])
    try:
        capacity = max(int(enclave.get("capacity") or 1), 1)
    except (TypeError, ValueError):
        capacity = 1
    return max(reported, capacity * per_job, 1)
