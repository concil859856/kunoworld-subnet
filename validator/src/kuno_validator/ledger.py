"""Auditing the gateway's ledger before a single entry is paid.

The gateway is a relay, not a source of truth. Every succeeded entry must carry a
receipt that verifies against an enclave signing key the validator fetched and
checked itself; anything else is dropped and counted. Billable seconds come from
the job's public parameters (what the customer paid for), never from the miner's
own `receipt.video.duration_s`. A content digest that shows up in more than one
job is a replay: only its first delivery earns.

Replay policy (also in VALIDATING.md):
  * the earliest verified delivery of a digest (by finished_at, then job_id) is credited;
  * every later delivery earns nothing;
  * a later delivery by a *different* miner than the first zeroes that miner for the
    scoring window, since it can only come from copying another miner's output;
  * a repeat by the *same* miner is flagged but not penalized beyond earning nothing,
    because an identical resubmission can legitimately reproduce identical bytes.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from pydantic import ValidationError

from kuno_protocol.attestation import enclave_id_for
from kuno_protocol.canonical import b64d, canonical_json, sha256_hex
from kuno_protocol.profiles import ModelProfile
from kuno_protocol.receipts import Receipt, verify_receipt
from kuno_protocol.schemas import GenerationParams

log = logging.getLogger("kuno.validator.ledger")

# How far the rendered length may sit from the requested one. Models render on a frame
# grid (H3: 17n+5 frames at 24 fps; LTX: 8k+1), so the real clip can overshoot slightly.
DURATION_SLACK_S = 0.5


@dataclass
class EnclaveKey:
    enclave_id: str
    miner_hotkey: str | None
    signing_public_key: bytes


def enclave_keys(enclaves: list[dict]) -> dict[str, EnclaveKey]:
    """Signing keys from the gateway's enclave list, kept only when self-certifying.

    An enclave id is a hash of its two public keys, so the gateway cannot swap in a
    key of its own without the id changing. Entries that fail that check are ignored.
    """
    keys: dict[str, EnclaveKey] = {}
    for enclave in enclaves:
        try:
            hpke, signing = b64d(enclave["hpke_public_key"]), b64d(enclave["signing_public_key"])
        except (KeyError, TypeError, ValueError):
            log.warning("enclave entry without usable keys ignored")
            continue
        if enclave_id_for(hpke, signing) != enclave.get("enclave_id"):
            log.warning("enclave %s: keys do not hash to its id; ignored", enclave.get("enclave_id"))
            continue
        keys[enclave["enclave_id"]] = EnclaveKey(enclave["enclave_id"], enclave.get("miner_hotkey"), signing)
    return keys


def duration_bounds(profile: ModelProfile, duration_s: float, fps: int | None) -> tuple[float, float]:
    """Acceptable rendered length for a requested duration, allowing for the model's frame grid."""
    fps = fps or profile.limits.default_fps
    rendered = profile.num_frames(duration_s, fps) / fps
    return duration_s - DURATION_SLACK_S, max(duration_s, rendered) + DURATION_SLACK_S


@dataclass
class LedgerAudit:
    """What survived verification, and why the rest did not."""

    entries: list[dict] = field(default_factory=list)
    dropped: Counter = field(default_factory=Counter)
    # hotkey -> human-readable flags (duration mismatches, same-miner repeats, unbound params)
    flags: dict[str, list[str]] = field(default_factory=dict)
    # hotkey -> reasons that disqualify the miner for the window (cross-miner replay)
    penalties: dict[str, list[str]] = field(default_factory=dict)
    # Entries whose duration came from the gateway's `duration_s` rather than full params
    # bound to the signed params digest. Scored, but only as trustworthy as the gateway.
    unbound: int = 0

    def flag(self, hotkey: str, message: str) -> None:
        self.flags.setdefault(hotkey, []).append(message)

    @property
    def dropped_total(self) -> int:
        return sum(self.dropped.values())


def audit_ledger(ledger: list[dict], keys: dict[str, EnclaveKey], profiles: dict[str, ModelProfile]) -> LedgerAudit:
    """Returns entries safe to score. Input dicts are never mutated.

    Each surviving succeeded entry gains `billable_s` (seconds from public params) and
    `credit` (False when it must not earn: a duration mismatch or a replay).
    Failed entries pass through untouched; they carry no receipt to verify.
    """
    audit = LedgerAudit()
    seen_jobs: set[str] = set()
    verified: list[tuple[dict, Receipt]] = []
    for raw in ledger:
        job_id = raw.get("job_id")
        if job_id is not None:
            if job_id in seen_jobs:
                audit.dropped["duplicate ledger row"] += 1
                continue
            seen_jobs.add(job_id)
        if raw.get("status") != "succeeded":
            audit.entries.append(dict(raw))
            continue
        outcome = _verify_entry(raw, keys, profiles)
        if isinstance(outcome, str):
            audit.dropped[outcome] += 1
            continue
        entry, receipt = outcome
        if entry.pop("_params_unbound", False):
            audit.unbound += 1
        verified.append((entry, receipt))

    _mark_replays(verified, audit)
    audit.entries.extend(entry for entry, _ in verified)
    audit.entries.sort(key=lambda e: (e.get("finished_at") or 0, e.get("job_id") or ""))
    if audit.dropped_total:
        log.warning(
            "dropped %d unverifiable ledger entries: %s",
            audit.dropped_total, ", ".join(f"{reason}={n}" for reason, n in audit.dropped.items()),
        )
    if audit.unbound:
        log.info("%d entries billed from the gateway's duration_s (ledger has no params to bind)", audit.unbound)
    return audit


def _verify_entry(raw: dict, keys: dict[str, EnclaveKey], profiles: dict[str, ModelProfile]) -> tuple[dict, Receipt] | str:
    """A verified copy of the entry and its receipt, or the reason it was dropped."""
    if not raw.get("receipt"):
        return "succeeded without a receipt"
    try:
        receipt = Receipt.model_validate(raw["receipt"])
    except ValidationError:
        return "malformed receipt"
    body = receipt.body
    key = keys.get(body.enclave_id)
    if key is None:
        return "receipt from an unknown enclave"
    if not verify_receipt(receipt, key.signing_public_key):
        return "bad receipt signature"
    if (
        body.job_id != raw.get("job_id")
        or body.profile_id != raw.get("profile_id")
        or (raw.get("enclave_id") is not None and body.enclave_id != raw["enclave_id"])
    ):
        return "receipt does not match its ledger entry"
    if key.miner_hotkey is None or (body.miner_hotkey is not None and body.miner_hotkey != key.miner_hotkey):
        return "receipt hotkey does not match the enclave's registered miner"
    profile = profiles.get(body.profile_id)
    if profile is None:
        return "unknown profile"

    entry = dict(raw, miner_hotkey=key.miner_hotkey)
    fps: int | None = None
    if raw.get("params") is not None:
        # Preferred: the gateway publishes the full public params, which we bind to the signed digest.
        try:
            params = GenerationParams.model_validate(raw["params"])
        except ValidationError:
            return "malformed params"
        if sha256_hex(canonical_json(params.model_dump(mode="json"))) != body.params_digest:
            return "params do not match the signed params digest"
        if params.profile_id != body.profile_id:
            return "receipt does not match its ledger entry"
        requested, fps = params.duration_s, params.fps
    elif isinstance(raw.get("duration_s"), (int, float)):
        requested = float(raw["duration_s"])
        entry["_params_unbound"] = True
    else:
        return "no public duration for the job"

    low, high = duration_bounds(profile, requested, fps)
    entry["billable_s"] = requested
    entry["credit"] = True
    claimed = body.video.duration_s
    if not low <= claimed <= high:
        entry["credit"] = False
        entry["flag"] = f"job {body.job_id}: receipt reports {claimed:g}s for a {requested:g}s request"
    entry["content_digest"] = body.content_digest
    return entry, receipt


def _mark_replays(verified: list[tuple[dict, Receipt]], audit: LedgerAudit) -> None:
    first: dict[str, dict] = {}
    for entry, _receipt in sorted(verified, key=lambda pair: (pair[0].get("finished_at") or 0, pair[0]["job_id"])):
        hotkey = entry["miner_hotkey"]
        if "flag" in entry:
            audit.flag(hotkey, entry.pop("flag"))
        digest = entry["content_digest"]
        original = first.get(digest)
        if original is None:
            first[digest] = entry
            continue
        entry["credit"] = False
        entry["replay_of"] = original["job_id"]
        if original["miner_hotkey"] != hotkey:
            audit.penalties.setdefault(hotkey, []).append(
                f"replayed output of job {original['job_id']} (another miner) as job {entry['job_id']}"
            )
        else:
            audit.flag(hotkey, f"job {entry['job_id']} repeats the output of its own job {original['job_id']}")
