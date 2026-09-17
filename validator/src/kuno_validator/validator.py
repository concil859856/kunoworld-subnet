from __future__ import annotations

import json
import logging
import math
import os
import random
import secrets
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import httpx

from kuno_protocol.attestation import AttestationEvidence, AttestationPolicy, GoldenManifest, Verdict, enclave_id_for, verify_endorsed_evidence
from kuno_protocol.canonical import b64d, b64e, canonical_json, sha256_hex
from kuno_protocol.crypto import DecryptionError, SenderSession
from kuno_protocol.findings import MAX_FINDINGS, Finding, FindingsReport, SignedFindings, sign_findings, verify_findings
from kuno_protocol.hotkey import HotkeySigner
from kuno_protocol.location import LandmarkList, LocationProof, SignedLandmarks, verify_location
from kuno_protocol.mp4 import Mp4Error, probe
from kuno_protocol.plans import PLAN_FEATURE, PLAN_OPTION, Plan, PlanError, PlanOptions, open_plan
from kuno_protocol.profiles import Mode, ModelProfile, load_profiles
from kuno_protocol.receipts import Receipt, verify_receipt
from kuno_protocol.schemas import GenerationParams, JobCreate, JobState, JobStatus, RouteResponse, SealedPayload, job_aad
from kuno_protocol.sealed_payload import seal_payload
from kuno_protocol.switch import SignedSwitch, SwitchConfig
from kuno_protocol.tiers import OPEN, tier_for_tee
from kuno_protocol.tolerance import Calibration
from kuno_protocol.turbo import is_candidate_profile_list

from .audits import AuditOutcome, AuditPolicy, Auditor, CanaryRecord
from .canaries import pick_prompt
from .capacity import CapacityTracker
from .collateral import CollateralGate
from .open_tier import AdmissionTracker, TierPolicy, apply_tiers, fraud_penalties, open_tier_gpus
from .plan_canaries import PlanBrief, check_plan, pick_brief
from .ledger import DURATION_SLACK_S, EnclaveKey, LedgerAudit, audit_ledger, duration_bounds, enclave_keys
from .scoring import CapacityCredit, MinerScore, compute_scores, hardware_conflicts, normalize
from .usd_pay import UsdPay

log = logging.getLogger("kuno.validator")

LEDGER_PAGE = 5000
# Canary outcomes older than this are pruned from the state file; it must exceed any scoring window.
CANARY_RETENTION_S = 7 * 86400.0
# How often a standard canary polls its job.
STANDARD_CANARY_POLL_S = 2.0
# Validator roles (VALIDATING.md, "Validator roles"): the main validator tests miners; auditors audit.
Role = Literal["main", "auditor"]
ROLES: tuple[str, ...] = ("main", "auditor")
# The share of active enclaves an auditor challenges with its own nonce each round, on top of the published evidence.
DEFAULT_SPOT_CHECK_RATE = 0.1
# Above this share of weight moved, an auditor warns that its weights and the main validator's disagree.
DEFAULT_DIVERGENCE_WARNING = 0.1


def weight_divergence(mine: dict[str, float], theirs: dict[str, float]) -> float:
    """Half the L1 distance between two normalized weight vectors: the share of weight that would have to move, 0..1."""
    def unit(weights: dict[str, float]) -> dict[str, float]:
        total = sum(w for w in weights.values() if w > 0)
        return {k: w / total for k, w in weights.items() if w > 0} if total > 0 else {}

    a, b = unit(mine), unit(theirs)
    if not a and not b:
        return 0.0
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


class GatewayAuthError(RuntimeError):
    """The gateway refused the validator API key."""


@dataclass
class CanaryResult:
    """One canary outcome.

    `attributable` is True only when the failure is provably the miner's: the receipt
    verified against that miner's enclave key, so the gateway cannot have framed it.
    Only attributable failures cost weight (see VALIDATING.md, "Canary policy").
    """

    profile_id: str
    ok: bool
    detail: str
    job_id: str | None = None
    enclave_id: str | None = None
    miner_hotkey: str | None = None
    attributable: bool = False
    at: float = 0.0


class Validator:
    def __init__(
        self,
        gateway_url: str,
        api_key: str,
        manifest: GoldenManifest,
        owner_public_key: bytes | None = None,
        transport: httpx.BaseTransport | None = None,
        country: str | None = None,
        state_path: Path | None = None,
        policy: AttestationPolicy | None = None,
        collateral: CollateralGate | None = None,
        tier_policy: TierPolicy | None = None,
        calibration: Calibration | None = None,
        pay: UsdPay | None = None,
        capacity: CapacityTracker | None = None,
        role: Role = "main",
        findings_signer: HotkeySigner | None = None,
        main_validator_hotkey: str | None = None,
        spot_check_rate: float = DEFAULT_SPOT_CHECK_RATE,
        divergence_warning: float = DEFAULT_DIVERGENCE_WARNING,
        require_location_proof: bool = False,
        plan_briefs: list[PlanBrief] | None = None,
    ):
        if role not in ROLES:
            raise ValueError(f"validator role must be one of {', '.join(ROLES)}, not {role!r}")
        if not 0.0 <= spot_check_rate <= 1.0:
            raise ValueError("the spot-check rate is a share of enclaves, between 0 and 1")
        if not api_key:
            raise ValueError("a validator API key is required: the gateway authenticates every validator read")
        self.gateway_url = gateway_url.rstrip("/")
        self.manifest = manifest
        # The same policy the gateway runs: real TDX and GPU verifiers, and production's stricter rules.
        self.policy = policy or AttestationPolicy()
        # The Turbo track (mechanism 1), when this validator runs one; its benchmarks aren't serving work.
        self.turbo = None
        self.owner_public_key = owner_public_key
        self.profiles = load_profiles()
        self._http = httpx.Client(
            base_url=self.gateway_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0, transport=transport
        )
        # Canaries for region-licensed models (H3) must originate from a licensed region.
        self._sdk_args = (api_key, country, transport)
        self._sdk = None
        self._hotkeys: dict[str, str] = {}
        self._keys: dict[str, EnclaveKey] = {}
        self._switch: SignedSwitch | None = None
        self.canary_results: list[CanaryResult] = []
        self.last_audit: LedgerAudit | None = None
        # Stake behind attested GPUs (collateral.py); None when no requirement is configured.
        self.collateral = collateral
        # token -> {"kind", "hotkeys": {hotkey: [first_seen, last_seen]}} from our own verdicts only.
        self.hardware_sightings: dict[str, dict] = {}
        self._feed_hardware: dict[str, set[str]] = {}
        # Open tier (open_tier.py): earning rate, admission probes, and the tiers this validator verified itself.
        self.tier_policy = tier_policy or TierPolicy()
        self.admission = AdmissionTracker(self.tier_policy.admission_probes)
        self.enclave_tiers: dict[str, str] = {}
        self._feed: dict[str, dict] = {}
        # USD-denominated pay (usd_pay.py) when KUNO_PAY_MODE=usd; None keeps today's VCU scoring.
        self.pay = pay
        self._stored_rate_card: dict | None = None
        # (now, window_s, switch) of the last score(), so USD pay prices exactly the round that was scored.
        self._scored: tuple[float, float, SwitchConfig] | None = None
        # Capacity pay (capacity.py): verified runs of confidential-tier GPUs, the profiles each enclave's evidence
        # claimed in our own challenges this round, and the capacity credit of the last score().
        self.capacity = capacity or CapacityTracker()
        self.enclave_profiles: dict[str, list[str]] = {}
        self.last_capacity: CapacityCredit | None = None
        # The main validator signs its findings with its hotkey; an auditor applies only findings its configured main
        # validator signed, and compares its weights with the ones that validator published.
        self.role: Role = role
        self.findings_signer = findings_signer
        self.main_validator_hotkey = main_validator_hotkey
        self.spot_check_rate = spot_check_rate
        self.divergence_warning = divergence_warning
        self.last_divergence: float | None = None
        # Territory-bound profiles (MiniMax H3) count as attested only with a landmark proof this validator checked.
        self.require_location_proof = require_location_proof
        # Plan canaries' briefs (plan_canaries.py); None: the public fallback set.
        self.plan_briefs = plan_briefs
        self._landmark_list: LandmarkList | None = None
        self._main_weights: dict[str, float] | None = None
        self.state_path = state_path
        self._load_state()
        # Step audits of this validator's own canaries: open a random denoising step and replay it (VERIFIED_MODE.md).
        audit_state = state_path.with_name(f"{state_path.stem}-audits.json") if state_path is not None else None
        self.auditor = Auditor(
            self._request, self.profiles, lambda enclave_id: self._keys.get(enclave_id),
            policy=AuditPolicy(production=self.policy.production), state_path=audit_state, calibration=calibration,
        )
        self._unaudited: list[CanaryRecord] = []
        if owner_public_key is None:
            log.error(
                "NO OWNER PUBLIC KEY CONFIGURED: the model switch cannot be verified and a compromised gateway "
                "can redirect emissions. Set KUNO_OWNER_PUBLIC_KEY before setting weights."
            )

    @property
    def sdk(self):
        """The client SDK, imported lazily: scoring and attestation don't need it."""
        if self._sdk is None:
            from kunoworld import KunoClient

            api_key, country, transport = self._sdk_args
            self._sdk = KunoClient(api_key, self.gateway_url, manifest=self.manifest, country=country, transport=transport)
        return self._sdk

    def close(self) -> None:
        self._http.close()
        if self._sdk is not None:
            self._sdk.close()

    # ------------------------------------------------------------ gateway

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Every gateway call goes through here, so every call carries the API key."""
        response = self._http.request(method, path, **kwargs)
        if response.status_code in (401, 403):
            raise GatewayAuthError(f"gateway rejected the validator API key for {method} {path} ({response.status_code})")
        return response

    # ------------------------------------------------------------ inputs

    def switch(self) -> SwitchConfig:
        """The owner-signed switch, accepted only if verified and not older than the last one."""
        signed = SignedSwitch.model_validate(self._request("GET", "/v1/switch").raise_for_status().json())
        if self.owner_public_key is None:
            log.error("using an UNVERIFIED model switch: no owner public key is configured")
        elif not signed.verify(self.owner_public_key):
            log.warning("gateway switch is not signed by the owner key; keeping the last verified switch")
            return self._accepted_switch()
        last = self._switch
        if last is not None:
            if signed.config.issued_at < last.config.issued_at:
                log.warning(
                    "gateway served a switch issued at %d, older than the accepted %d; ignoring the rollback",
                    signed.config.issued_at, last.config.issued_at,
                )
                return last.config
            if signed.config.issued_at == last.config.issued_at and signed.config != last.config:
                log.warning("gateway served a different switch with the same issued_at; keeping the accepted one")
                return last.config
        if last is None or signed != last:
            self._switch = signed
            self._save_state()
        return signed.config

    def _accepted_switch(self) -> SwitchConfig:
        return self._switch.config if self._switch is not None else SwitchConfig()

    def enclaves(self) -> list[dict]:
        """All enclaves the gateway knows, refreshing the self-certified signing keys."""
        enclaves = self._request("GET", "/validator/v1/enclaves").raise_for_status().json()
        self._hotkeys = {e["enclave_id"]: e["miner_hotkey"] for e in enclaves}
        self._feed = {e["enclave_id"]: e for e in enclaves}
        # What the gateway says each enclave's hardware is. Only compared against, never scored on.
        self._feed_hardware = {e["enclave_id"]: {h["token"] for h in e.get("hardware_ids") or []} for e in enclaves}
        # Keys are bound to their ids, so merging keeps retired enclaves' receipts verifiable.
        self._keys.update(enclave_keys(enclaves))
        return enclaves

    def ledger(self, since: float) -> list[dict]:
        rows: list[dict] = []
        while True:
            page = self._request("GET", "/validator/v1/ledger", params={"since": since, "limit": LEDGER_PAGE}).raise_for_status().json()
            rows.extend(page)
            if len(page) < LEDGER_PAGE:
                return rows
            since = max(row.get("finished_at") or since for row in page)

    # ------------------------------------------------------------ attestation

    def _active_enclaves(self) -> list[dict]:
        # Turbo candidates are challenged by the Turbo track against their own manifest.
        return [
            e for e in self.enclaves()
            if e["status"] == "active" and not is_candidate_profile_list(list(e.get("profiles") or []))
        ]

    def check_enclaves(self, timeout_s: float = 30.0, enclaves: list[dict] | None = None) -> dict[str, Verdict]:
        """Challenges every active enclave (or those given) with our own nonce and verifies the answer ourselves."""
        if enclaves is None:
            enclaves = self._active_enclaves()
            self.enclave_profiles = {}
        pending: dict[str, tuple[dict, bytes]] = {}
        for enclave in enclaves:
            nonce = os.urandom(32)
            response = self._request("POST", "/validator/v1/challenges", json={"enclave_id": enclave["enclave_id"], "nonce": nonce.hex()})
            if response.status_code == 201:
                pending[response.json()["challenge_id"]] = (enclave, nonce)

        verdicts: dict[str, Verdict] = {}
        deadline = time.time() + timeout_s
        while pending and time.time() < deadline:
            for challenge_id, (enclave, nonce) in list(pending.items()):
                answer = self._request("GET", f"/validator/v1/challenges/{challenge_id}").json()
                if answer["status"] == "answered" and answer["evidence"]:
                    evidence = AttestationEvidence.model_validate(answer["evidence"])
                    # Open-tier enclaves answer with open evidence, accepted only where our manifest's open_tier policy allows it.
                    # Confidential evidence is verified exactly as before, so policies without `allow_open` keep working.
                    extra = {"allow_open": True} if evidence.tee == "open" else {}
                    verdict = self.policy.verify(evidence, self.manifest, expected_nonce=nonce, **extra)
                    if verdict.enclave_id != enclave["enclave_id"]:
                        verdict.ok = False
                        verdict.reasons.append("answered with different keys than the registered enclave")
                    else:
                        # Only verdicts this validator reached decide an enclave's tier (rates, admission, the fraud
                        # rule), and the profiles capacity pay splits GPU-time by.
                        self._accept_verdict(enclave, evidence, verdict)
                    verdicts[enclave["enclave_id"]] = verdict
                    del pending[challenge_id]
                elif answer["status"] == "expired":
                    del pending[challenge_id]
            if pending:
                time.sleep(0.5)
        for challenge_id, (enclave, _) in pending.items():
            verdicts[enclave["enclave_id"]] = Verdict(False, enclave["enclave_id"], ["did not answer the challenge in time"])
        return verdicts

    def published_verdicts(self, spot_timeout_s: float = 30.0) -> dict[str, Verdict]:
        """An auditor's attestation check: every active enclave's published evidence, verified here, plus challenges
        with our own nonce for a random `spot_check_rate` share of them.

        Published evidence carries someone else's nonce, so on its own it proves only that the enclave was genuine
        within the manifest's `max_evidence_age_s`, which the gateway keeps fresh by replacing it at every
        re-attestation. The spot challenges are what catch a gateway serving evidence an enclave can no longer
        produce. A TDX quote is checked with this validator's own DCAP and NVIDIA verifiers when it has them
        (production requires them), else with the Intel and NVIDIA material relayed next to the evidence
        (kuno_protocol.endorsements), so no step takes the gateway's word.
        """
        enclaves = self._active_enclaves()
        self.enclave_profiles = {}
        verdicts: dict[str, Verdict] = {}
        for enclave in enclaves:
            enclave_id = enclave["enclave_id"]
            published = enclave.get("evidence")
            if not published:
                verdicts[enclave_id] = Verdict(False, enclave_id, ["the gateway publishes no evidence for this enclave"])
                continue
            try:
                evidence = AttestationEvidence.model_validate(published)
            except ValueError:
                verdicts[enclave_id] = Verdict(False, enclave_id, ["the published evidence is malformed"])
                continue
            if evidence.tee == "tdx" and self.policy.quote_verifier is None:
                verdict = verify_endorsed_evidence(evidence, self.manifest, enclave.get("endorsements"))
            else:
                extra = {"allow_open": True} if evidence.tee == "open" else {}
                verdict = self.policy.verify(evidence, self.manifest, **extra)
            self._accept_verdict(enclave, evidence, verdict)
            verdicts[enclave_id] = verdict
        spot = [e for e in enclaves if verdicts.get(e["enclave_id"]) is not None and verdicts[e["enclave_id"]].ok]
        count = math.ceil(self.spot_check_rate * len(spot)) if spot else 0
        if count:
            chosen = random.SystemRandom().sample(spot, count)
            answered = self.check_enclaves(spot_timeout_s, chosen)
            for enclave in chosen:
                enclave_id = enclave["enclave_id"]
                # A challenge that expired unanswered leaves no verdict: for a spot check that is a failure.
                verdict = answered.get(enclave_id) or Verdict(False, enclave_id, ["did not answer our spot challenge"])
                if not verdict.ok:
                    log.warning("enclave %s passed on its published evidence but failed our spot challenge: %s",
                                enclave_id, "; ".join(verdict.reasons))
                    self.enclave_profiles.pop(enclave_id, None)
                verdicts[enclave_id] = verdict
        return verdicts

    def _accept_verdict(self, enclave: dict, evidence: AttestationEvidence, verdict: Verdict) -> None:
        if verdict.enclave_id != enclave["enclave_id"]:
            verdict.ok = False
            verdict.reasons.append("the evidence names different keys than the registered enclave")
            return
        problem = self.location_problem(enclave, evidence.profiles) if verdict.ok else None
        if problem:
            verdict.ok = False
            verdict.reasons.append(problem)
        elif verdict.ok and verdict.tier:
            self.enclave_tiers[verdict.enclave_id] = verdict.tier
        if verdict.ok:
            self.enclave_profiles[verdict.enclave_id] = list(evidence.profiles)

    def landmarks(self) -> LandmarkList | None:
        """The gateway's landmark list, used only if the owner signed it (or, with no owner key, as served)."""
        response = self._request("GET", "/v1/landmarks")
        if response.status_code != 200:
            return None
        try:
            signed = SignedLandmarks.model_validate(response.json())
        except ValueError:
            log.error("the gateway served a malformed landmark list")
            return None
        if self.owner_public_key is not None and not signed.verify(self.owner_public_key):
            log.error("the gateway's landmark list is not signed by the owner key; location proofs can't be checked")
            return None
        return signed.landmarks

    def location_problem(self, enclave: dict, profiles: list[str]) -> str | None:
        """Why an enclave offering territory-bound profiles hasn't proven where it runs, or None (kuno_protocol.location)."""
        policies = {self.profiles[p].license.region_policy for p in profiles if p in self.profiles} - {None}
        if not policies or not self.require_location_proof:
            return None
        if self._landmark_list is None:
            self._landmark_list = self.landmarks()
        if self._landmark_list is None:
            return "no owner-signed landmark list to check this enclave's location against"
        record = enclave.get("location") or {}
        try:
            proof = LocationProof.model_validate(record["proof"]) if record.get("proof") else None
        except ValueError:
            proof = None
        for policy in sorted(policies):
            verdict = verify_location(proof, self._landmark_list, registration_nonce=str(record.get("nonce") or ""),
                                      enclave_id=enclave["enclave_id"], region_policy=policy)
            if not verdict.ok:
                return f"location not proven for {policy}: {verdict.detail}"
        return None

    # ------------------------------------------------------------ findings (main validator -> auditors)

    def findings(self, now: float, window_s: float) -> list[Finding]:
        """The main validator's attributable failures in the window: its canaries and its step audits."""
        out = [
            Finding(kind="canary_failed", miner_hotkey=r.miner_hotkey, detail=r.detail[:500], at=r.at, job_id=r.job_id,
                    enclave_id=r.enclave_id, profile_id=r.profile_id)
            for r in self.canary_results
            if not r.ok and r.attributable and r.miner_hotkey and r.at >= now - window_s
        ]
        out += [
            Finding(kind="audit_failed", miner_hotkey=o.miner_hotkey, detail=o.detail[:500], at=o.at, job_id=o.job_id,
                    enclave_id=o.enclave_id, profile_id=o.profile_id)
            for o in self.auditor.outcomes
            if not o.ok and o.attributable and o.miner_hotkey and o.at >= now - window_s
        ]
        return sorted(out, key=lambda f: f.at, reverse=True)[:MAX_FINDINGS]

    def publish_findings(self, weights: dict[str, float], now: float, window_s: float) -> bool:
        """Signs this round's findings and weights with the main validator's hotkey and hands them to the gateway."""
        if self.findings_signer is None:
            log.error("no hotkey to sign findings with: auditors can't see this validator's canary and audit failures")
            return False
        report = FindingsReport(
            validator_hotkey=self.findings_signer.ss58_address, issued_at=now, window_s=window_s,
            findings=self.findings(now, window_s), weights={k: round(v, 9) for k, v in weights.items()},
        )
        response = self._request("POST", "/validator/v1/findings", json=sign_findings(self.findings_signer, report).model_dump(mode="json"))
        if response.status_code not in (200, 201):
            log.error("the gateway refused this round's findings (%d): %s", response.status_code, response.text[:200])
            return False
        return True

    def main_validator_findings(self, now: float, window_s: float) -> dict[str, list[str]]:
        """An auditor's penalties from the main validator: findings in the window from reports its hotkey signed.

        Also keeps the weights of the newest verified report, which `step` compares with this validator's own."""
        if self.main_validator_hotkey is None:
            log.error("no main validator hotkey configured: this auditor applies none of its canary or audit findings")
            return {}
        response = self._request("GET", "/validator/v1/findings", params={"since": now - window_s})
        if response.status_code != 200:
            log.error("could not read the main validator's findings (%d); scoring without them", response.status_code)
            return {}
        penalties: dict[str, list[str]] = {}
        seen: set[tuple[str, str | None, str]] = set()
        newest: FindingsReport | None = None
        for document in response.json():
            try:
                signed = SignedFindings.model_validate(document)
            except ValueError:
                log.warning("skipping a malformed findings report from the gateway")
                continue
            ok, detail = verify_findings(signed, self.main_validator_hotkey)
            if not ok:
                log.warning("skipping a findings report: %s", detail)
                continue
            report = signed.report
            if newest is None or report.issued_at > newest.issued_at:
                newest = report
            for finding in report.findings:
                key = (finding.kind, finding.job_id, finding.miner_hotkey)
                if finding.at < now - window_s or key in seen:
                    continue
                seen.add(key)
                noun = "canary" if finding.kind == "canary_failed" else "step audit of"
                penalties.setdefault(finding.miner_hotkey, []).append(
                    f"failed {noun} {finding.profile_id or 'a job'} ({finding.detail}) [main validator]"
                )
        self._main_weights = dict(newest.weights) if newest is not None and newest.weights is not None else None
        return penalties

    def attested_hotkeys(self, verdicts: dict[str, Verdict]) -> set[str]:
        return {self._hotkeys[eid] for eid, v in verdicts.items() if v.ok and self._hotkeys.get(eid)}

    # ------------------------------------------------------------ hardware and collateral

    def record_hardware(self, verdicts: dict[str, Verdict], now: float, window_s: float) -> None:
        """Remembers which hotkey showed which verified hardware, from our own challenge verdicts.

        A hotkey not seen with a token for a whole window starts over, so after hardware changes
        hands the earlier owner stops counting once the window passes it.
        """
        for enclave_id, verdict in verdicts.items():
            hotkey = self._hotkeys.get(enclave_id)
            if not verdict.ok or not hotkey:
                continue
            feed = self._feed_hardware.get(enclave_id)
            if feed and feed != verdict.hardware_tokens():
                log.warning("gateway feed lists different hardware for enclave %s than its challenge answer attested", enclave_id)
            for identity in verdict.hardware:
                entry = self.hardware_sightings.setdefault(identity.token, {"kind": identity.kind, "hotkeys": {}})
                previous = entry["hotkeys"].get(hotkey)
                first = previous[0] if previous and previous[1] >= now - window_s else now
                entry["hotkeys"][hotkey] = [first, now]
        for token in list(self.hardware_sightings):
            hotkeys = self.hardware_sightings[token]["hotkeys"]
            for hotkey in [h for h, (_, last) in hotkeys.items() if last < now - window_s]:
                del hotkeys[hotkey]
            if not hotkeys:
                del self.hardware_sightings[token]

    def attested_gpus(self, verdicts: dict[str, Verdict]) -> dict[str, int]:
        """GPUs each hotkey has attested this round, each GPU counted once however many enclaves show it."""
        named: dict[str, set[str]] = {}
        unnamed: dict[str, int] = {}
        for enclave_id, verdict in verdicts.items():
            hotkey = self._hotkeys.get(enclave_id)
            if not verdict.ok or not hotkey or verdict.tier == OPEN:
                continue
            gpus = verdict.hardware_tokens("gpu")
            named.setdefault(hotkey, set()).update(gpus)
            # GPUs counted without an identity still need collateral, and every enclave at least one GPU's worth.
            extra = max((verdict.gpu_count or 0) - len(gpus), 0)
            unnamed[hotkey] = unnamed.get(hotkey, 0) + (extra if gpus or extra else 1)
        return {hotkey: len(named[hotkey]) + unnamed.get(hotkey, 0) for hotkey in named}

    def open_tier_gpus(self, verdicts: dict[str, Verdict]) -> dict[str, int]:
        """GPUs each hotkey runs on the open tier this round: reported or capacity-derived, never attested."""
        needs = {profile_id: profile.gpus_per_worker for profile_id, profile in self.profiles.items()}
        counts: dict[str, int] = {}
        for enclave_id, verdict in verdicts.items():
            hotkey = self._hotkeys.get(enclave_id)
            if verdict.ok and hotkey and verdict.tier == OPEN:
                counts[hotkey] = counts.get(hotkey, 0) + open_tier_gpus(self._feed.get(enclave_id, {}), needs)
        return counts

    def record_capacity(self, verdicts: dict[str, Verdict], now: float, window_s: float) -> None:
        """Checks for capacity pay (capacity.py): every GPU identity our own challenges verified on the confidential tier.

        Open-tier GPUs are never attested, only reported, and GPUs counted without an identity can't be told apart,
        so neither is checked. Gates are applied when the credit is scored, like any other work in the window.
        """
        checks: list[tuple[str, str, list[str]]] = []
        for enclave_id, verdict in verdicts.items():
            hotkey = self._hotkeys.get(enclave_id)
            if not verdict.ok or not hotkey or verdict.tier == OPEN:
                continue
            profiles = self.enclave_profiles.get(enclave_id)
            if profiles is None:  # a verdict from outside check_enclaves (tools, tests): the feed's profile list
                profiles = list(self._feed.get(enclave_id, {}).get("profiles") or [])
            checks.extend((hotkey, token, profiles) for token in sorted(verdict.hardware_tokens("gpu")))
        self.capacity.record(checks, now, window_s)

    def served_families(self, profile_ids: list[str], switch: SwitchConfig) -> list[str]:
        """The families an enclave's GPU-time is split over: those of its profiles the switch has on."""
        return sorted({
            self.profiles[p].family for p in profile_ids if p in self.profiles and switch.profile_enabled(self.profiles[p])
        })

    def enclave_tier(self, enclave_id: str | None) -> str | None:
        """The tier this validator verified itself, else what the gateway's feed says; None when neither knows."""
        tier = self.enclave_tiers.get(enclave_id or "")
        if tier is None:
            feed = self._feed.get(enclave_id or "") or {}
            tier = feed.get("tier") or (tier_for_tee(feed["tee"]) if feed.get("tee") else None)
        return tier

    def known_tiers(self) -> dict[str, str]:
        """Enclave id -> tier for rates and admission: the feed, overridden by our own verdicts."""
        tiers = {eid: tier for eid in self._feed if (tier := self.enclave_tier(eid))}
        tiers.update(self.enclave_tiers)
        return tiers

    # ------------------------------------------------------------ canaries

    def run_canary(self, profile_id: str, privacy: str = "private") -> CanaryResult:
        if privacy == "standard":
            return self.run_standard_canary(profile_id)
        from kunoworld import KunoError  # only canaries need the client SDK

        profile = self.profiles[profile_id]
        prompt, seed, started = pick_prompt(), secrets.randbelow(2**31), time.time()
        try:
            result = self.sdk.generate(
                prompt,
                model=profile_id,
                seed=seed,
                duration_s=profile.limits.min_duration_s,
                resolution=next(iter(profile.limits.sizes)),
                timeout=profile.timeout_s,
            )
        except KunoError as exc:
            # No receipt, so nothing proves which miner is at fault; the ledger's reliability gate covers it.
            return self._record(CanaryResult(profile_id, False, exc.code))
        if result.profile_id != profile_id:
            detail = f"routed to {result.profile_id} ({result.fallback_reason}); canary did not test {profile_id}"
            return self._record(CanaryResult(profile_id, False, detail, job_id=result.job_id))
        outcome = self.check_canary_output(
            profile, result.job_id, result.video, result.receipt, profile.limits.min_duration_s, next(iter(profile.limits.sizes))
        )
        if profile.verified is not None:
            self._remember_canary(profile_id, result, prompt, seed, started)
        return self._record(outcome)

    def run_standard_canary(self, profile_id: str, sleep=time.sleep) -> CanaryResult:
        """A canary in standard mode (STANDARD_MODE.md): the gateway seals it, so it can reach open-tier miners.
        These are the admission probes an open-tier hotkey must pass before its work earns."""
        profile = self.profiles[profile_id]
        prompt, seed, started = pick_prompt(), secrets.randbelow(2**31), time.time()
        resolution = next(iter(profile.limits.sizes))
        params = GenerationParams(
            profile_id=profile.id, mode=Mode.TEXT_TO_VIDEO, duration_s=profile.limits.min_duration_s, resolution=resolution,
            aspect_ratio=next(iter(profile.limits.sizes[resolution])), fps=profile.limits.default_fps, audio=profile.limits.audio,
        )
        response = self._request("POST", "/v1/standard/videos", json={"params": params.model_dump(mode="json"), "prompt": prompt, "seed": seed})
        if response.status_code != 201:
            return self._record(CanaryResult(profile_id, False, f"standard canary refused ({response.status_code})"))
        status = JobStatus.model_validate(response.json())
        deadline = started + profile.timeout_s
        while not status.status.terminal and time.time() < deadline:
            sleep(STANDARD_CANARY_POLL_S)
            status = JobStatus.model_validate(self._request("GET", f"/v1/videos/{status.job_id}").raise_for_status().json())
        job_id = status.job_id
        if status.status != JobState.SUCCEEDED or status.receipt is None:
            # No receipt, so nothing proves which miner is at fault; the ledger's reliability gate covers it.
            return self._record(CanaryResult(profile_id, False, status.error_code or status.status.value, job_id=job_id))
        video = self._request("GET", f"/v1/standard/videos/{job_id}/video")
        if video.status_code != 200:
            return self._record(CanaryResult(profile_id, False, f"standard canary video unavailable ({video.status_code})", job_id=job_id))
        outcome = self.check_canary_output(profile, job_id, video.content, status.receipt, params.duration_s, resolution)
        signed = status.receipt.body.params_digest == sha256_hex(canonical_json(params.model_dump(mode="json")))
        if outcome.ok and not signed:
            outcome = CanaryResult(profile_id, False, "receipt signs different params than the canary requested", job_id,
                                   outcome.enclave_id, outcome.miner_hotkey, True)
        if outcome.ok and profile.verified is not None:
            self._unaudited.append(CanaryRecord(
                job_id=job_id, profile_id=profile_id, params=params.model_dump(mode="json"), prompt=prompt, seed=seed,
                receipt=status.receipt.model_dump(mode="json"), created_at=time.time(), tier=self.enclave_tier(outcome.enclave_id),
            ))
        return self._record(outcome)

    def run_plan_canary(self, profile_id: str, privacy: str = "private", sleep=time.sleep) -> CanaryResult:
        """A plan canary (plan_canaries.py). Private: sealed here to a confidential enclave that lists `plan/1` and whose
        attestation this validator verified, sent to `POST /v1/plans`, and opened with the job's output key. Standard:
        `POST /v1/standard/plans`, read back from `GET /v1/standard/plans/{job_id}`. Plans are never step-audited.
        A job that ends without a receipt, `plan_failed` included, proves nothing about which miner is at fault."""
        profile = self.profiles[profile_id]
        brief, seed = pick_brief(self.plan_briefs), secrets.randbelow(2**31)
        resolution = next(iter(profile.limits.sizes))
        sizes = profile.limits.sizes[resolution]
        params = GenerationParams(
            profile_id=profile.id, mode=Mode.PLAN, duration_s=brief.target_s, resolution=resolution,
            aspect_ratio="16:9" if "16:9" in sizes else next(iter(sizes)), fps=profile.limits.default_fps, audio=profile.limits.audio,
        )
        options = PlanOptions()
        output_key = None
        if privacy == "standard":
            response = self._request(
                "POST", "/v1/standard/plans", json={"params": params.model_dump(mode="json"), "brief": brief.brief, "seed": seed}
            )
        else:
            sealed = self._seal_plan(profile, params, brief.brief, seed, options)
            if isinstance(sealed, CanaryResult):
                return self._record(sealed)
            job, output_key = sealed
            response = self._request("POST", "/v1/plans", json=job.model_dump(mode="json"))
        if response.status_code != 201:
            return self._record(CanaryResult(profile_id, False, f"{privacy} plan canary refused ({response.status_code})"))
        status = JobStatus.model_validate(response.json())
        deadline = time.time() + profile.timeout_s
        while not status.status.terminal and time.time() < deadline:
            sleep(STANDARD_CANARY_POLL_S)
            status = JobStatus.model_validate(self._request("GET", f"/v1/videos/{status.job_id}").raise_for_status().json())
        job_id = status.job_id
        if status.status != JobState.SUCCEEDED or status.receipt is None:
            return self._record(CanaryResult(profile_id, False, f"plan: {status.error_code or status.status.value}", job_id=job_id))

        receipt = status.receipt
        body = receipt.body
        if body.enclave_id not in self._keys:
            self.enclaves()
        key = self._keys.get(body.enclave_id)
        if key is None or not verify_receipt(receipt, key.signing_public_key) or body.job_id != job_id:
            log.error("plan canary %s: the receipt does not verify for this job; the relay may be tampering", job_id)
            return self._record(CanaryResult(profile_id, False, "plan: receipt does not verify", job_id, body.enclave_id))

        def result(detail: str | None, attributable: bool = True) -> CanaryResult:
            ok = detail is None
            return self._record(CanaryResult(
                profile_id, ok, "ok" if ok else f"plan: {detail}", job_id, body.enclave_id, key.miner_hotkey, attributable,
            ))

        if body.profile_id != profile.id:
            return result(f"enclave signed a receipt for {body.profile_id}, not {profile.id}")
        if body.params_digest != sha256_hex(canonical_json(params.model_dump(mode="json"))):
            return result("receipt signs different params than the canary requested")
        if privacy == "standard":
            fetched = self._request("GET", f"/v1/standard/plans/{job_id}")
            # The gateway checked the plan against the receipt before keeping it, so a mismatch here is the gateway's.
            if fetched.status_code != 200 or sha256_hex(fetched.content) != body.content_digest:
                return result(f"the gateway serves no plan matching the receipt ({fetched.status_code})", attributable=False)
            plan_json = fetched.content
            try:
                plan = Plan.model_validate_json(plan_json)
            except ValueError:
                return result("the gateway serves a plan that doesn't parse", attributable=False)
        else:
            blob = self._request("GET", f"/v1/blobs/{status.output_blob_id}")
            if blob.status_code != 200 or sha256_hex(blob.content) != body.output_digest:
                return result(f"the output blob is missing or not the one the receipt certifies ({blob.status_code})", attributable=False)
            try:
                plan, plan_json = open_plan(output_key, job_id, blob.content)
            except (DecryptionError, PlanError):
                return result("the certified output does not open as a plan with the job's output key")
        return result(check_plan(plan, plan_json, receipt, profile, params, options, brief, self.manifest))

    def _seal_plan(
        self, profile: ModelProfile, params: GenerationParams, brief: str, seed: int, options: PlanOptions
    ) -> tuple[JobCreate, bytes] | CanaryResult:
        """A private plan job sealed exactly as a client seals one, to the first routed enclave that lists `plan/1` and
        whose attestation this validator checked; and the job's output key."""
        response = self._request("GET", "/v1/route", params={
            "mode": Mode.PLAN.value, "profile_id": profile.id, "resolution": params.resolution,
            "aspect_ratio": params.aspect_ratio, "fps": params.fps,
        })
        if response.status_code != 200:
            return CanaryResult(profile.id, False, f"plan canary not routed ({response.status_code})")
        route = RouteResponse.model_validate(response.json())
        if route.profile_id != profile.id:
            return CanaryResult(profile.id, False, f"plan canary routed to {route.profile_id} ({route.fallback_reason}); it did not test {profile.id}")
        enclave = self._attested_plan_enclave(route.enclaves, profile.id)
        if enclave is None:
            return CanaryResult(profile.id, False, "plan canary found no attested worker that writes plans")
        session = SenderSession(b64d(enclave["hpke_public_key"]))
        job_id = str(uuid.uuid4())
        payload = SealedPayload(prompt=brief, seed=seed, options={PLAN_OPTION: options.model_dump(mode="json", exclude_none=True)})
        ciphertext = seal_payload(session, payload, job_aad(job_id, enclave["enclave_id"], params, []))
        job = JobCreate(job_id=job_id, params=params, enclave_id=enclave["enclave_id"], enc=b64e(session.enc), ciphertext=b64e(ciphertext))
        return job, session.output_key

    def _attested_plan_enclave(self, enclaves: list[dict], profile_id: str) -> dict | None:
        """The first confidential enclave listing `plan/1` whose keys this validator trusts: attested by its own challenge
        this round, or else by verifying the evidence the route publishes, as auditors verify it."""
        for enclave in enclaves:
            enclave_id = enclave.get("enclave_id")
            try:
                keys = b64d(enclave["hpke_public_key"]), b64d(enclave["signing_public_key"])
            except (KeyError, TypeError, ValueError):
                continue
            if enclave_id_for(*keys) != enclave_id or PLAN_FEATURE not in (enclave.get("features") or []):
                continue
            if self.enclave_tiers.get(enclave_id) not in (None, OPEN) and profile_id in self.enclave_profiles.get(enclave_id, []):
                return enclave
            try:
                evidence = AttestationEvidence.model_validate(enclave["evidence"])
            except (KeyError, ValueError):
                continue
            if evidence.tee == "tdx" and self.policy.quote_verifier is None:
                verdict = verify_endorsed_evidence(evidence, self.manifest, enclave.get("endorsements"))
            else:
                verdict = self.policy.verify(evidence, self.manifest)
            if (
                verdict.ok and verdict.enclave_id == enclave_id and verdict.tier != OPEN and profile_id in evidence.profiles
                and evidence.hpke_public_key == enclave["hpke_public_key"]
            ):
                return enclave
        return None

    def _remember_canary(self, profile_id: str, result, prompt: str, seed: int, started: float) -> None:
        """Keeps what a step audit needs; the params come from the ledger so they hash to the signed digest."""
        params = next((row.get("params") for row in self.ledger(started - 60) if row.get("job_id") == result.job_id), None)
        if params is None:
            log.warning("canary %s is not in the ledger yet; it can't be step-audited", result.job_id)
            return
        self._unaudited.append(CanaryRecord(
            job_id=result.job_id, profile_id=profile_id, params=params, prompt=prompt, seed=seed,
            receipt=result.receipt.model_dump(mode="json"), created_at=time.time(),
        ))

    def standard_audit_records(self, now: float) -> list[CanaryRecord]:
        """Standard jobs sampled from the ledger, with the prompt and seed the gateway gives validators for them."""
        rows = self.ledger(now - self.auditor.policy.standard_max_age_s)
        records = []
        for row in self.auditor.sample_standard(rows, self.known_tiers(), now):
            response = self._request("GET", f"/validator/v1/standard-jobs/{row['job_id']}")
            if response.status_code != 200:
                log.info("standard job %s can't be audited: the gateway answered %s", row["job_id"], response.status_code)
                continue
            record = Auditor.standard_record(row, response.json())
            if record is None:
                log.info("standard job %s isn't replayable here (no explicit seed, inputs, options or a storyboard); not audited", row["job_id"])
                continue
            records.append(record)
        return records

    def run_audits(self) -> None:
        """Requests step audits for a sample of this round's canaries and of recent standard jobs, and waits for them."""
        outcomes: list[AuditOutcome] = []
        for canary in [*self.auditor.select(self._unaudited), *self.standard_audit_records(time.time())]:
            result = self.auditor.request(canary)
            if isinstance(result, AuditOutcome):
                outcomes.append(result)
        self._unaudited = []
        give_up = time.time() + self.auditor.policy.deadline_s + 5
        while self.auditor.pending and time.time() < give_up:
            outcomes.extend(self.auditor.poll())
            if self.auditor.pending:
                time.sleep(self.auditor.policy.poll_s)
        for outcome in outcomes:
            # An attributable audit failure during probation restarts an open-tier miner's admission.
            if outcome.attributable and not outcome.ok and outcome.miner_hotkey and outcome.tier == OPEN:
                self.admission.record(outcome.miner_hotkey, False, True, outcome.at)

    def check_canary_output(
        self, profile: ModelProfile, job_id: str, video: bytes, receipt: Receipt, duration_s: float, resolution: str
    ) -> CanaryResult:
        """Checks a delivered canary against the request, the receipt and the file itself."""

        def fail(detail: str, attributable: bool = True) -> CanaryResult:
            return CanaryResult(profile.id, False, detail, job_id, body.enclave_id, hotkey, attributable)

        body, hotkey = receipt.body, None
        if body.enclave_id not in self._keys:
            self.enclaves()
        key = self._keys.get(body.enclave_id)
        if key is None:
            return fail("receipt names an enclave the validator does not know", attributable=False)
        if not verify_receipt(receipt, key.signing_public_key):
            log.error("canary %s: receipt signature does not verify; the relay may be tampering", job_id)
            return fail("receipt signature does not verify", attributable=False)
        if body.job_id != job_id:
            return fail("receipt belongs to a different job", attributable=False)
        hotkey = key.miner_hotkey
        if body.profile_id != profile.id:
            return fail(f"enclave signed a receipt for {body.profile_id}, not {profile.id}")
        if body.video is None:
            return fail("enclave signed a plan receipt for a video canary")
        if sha256_hex(video) != body.content_digest:
            return fail("output does not match the receipt's content digest")
        try:
            info = probe(video)
        except Mp4Error as exc:
            return fail(f"output is not a playable MP4 ({exc})")
        low, high = duration_bounds(profile, duration_s, None)
        if not low <= info.duration_s <= high:
            return fail(f"rendered {info.duration_s:.2f}s for a {duration_s:g}s request")
        if abs(info.duration_s - body.video.duration_s) > DURATION_SLACK_S:
            return fail("receipt misreports the video duration")
        sizes = {tuple(size) for size in profile.limits.sizes.get(resolution, {}).values()}
        if (info.width, info.height) not in sizes:
            return fail(f"rendered {info.width}x{info.height}, not a {resolution} size")
        if (body.video.width, body.video.height) != (info.width, info.height):
            return fail("receipt misreports the video size")
        return CanaryResult(profile.id, True, "ok", job_id, body.enclave_id, hotkey, True)

    def _record(self, result: CanaryResult) -> CanaryResult:
        result.at = result.at or time.time()
        self.canary_results.append(result)
        if result.miner_hotkey and self.enclave_tier(result.enclave_id) == OPEN:
            self.admission.record(result.miner_hotkey, result.ok, result.attributable, result.at)
        self._save_state()
        return result

    def canary_penalties(self, now: float, window_s: float) -> dict[str, list[str]]:
        """Canary policy: any attributable failure inside the scoring window zeroes that miner."""
        penalties: dict[str, list[str]] = {}
        for result in self.canary_results:
            if result.ok or not result.attributable or not result.miner_hotkey or result.at < now - window_s:
                continue
            penalties.setdefault(result.miner_hotkey, []).append(f"failed canary {result.profile_id} ({result.detail})")
        return penalties

    # ------------------------------------------------------------ scoring

    def score(
        self, verdicts: dict[str, Verdict], window_s: float = 86400.0, extra_penalties: dict[str, list[str]] | None = None
    ) -> dict[str, MinerScore]:
        now = time.time()
        enclaves = self.enclaves()
        rows = self.ledger(now - window_s)
        if self.turbo is not None:
            from .turbo import exclude_benchmark_rows

            # Mechanism 1 already pays for Turbo benchmark jobs.
            rows = exclude_benchmark_rows(rows, enclaves, self.turbo.benchmark_job_ids())
        audit = audit_ledger(rows, self._keys, self.profiles)
        self.last_audit = audit
        penalties = {hotkey: list(reasons) for hotkey, reasons in audit.penalties.items()}
        # Tiers: open-tier work earns at the tier rate once admitted. The fraud rule trusts only our own verdicts.
        for enclave_id, verdict in verdicts.items():
            if verdict.ok and verdict.tier:
                self.enclave_tiers[enclave_id] = verdict.tier
        flags = {hotkey: list(items) for hotkey, items in audit.flags.items()}
        for hotkey, items in apply_tiers(audit.entries, self.known_tiers(), self.admission).items():
            flags.setdefault(hotkey, []).extend(items)
        for hotkey, reasons in fraud_penalties(audit.entries, self.enclave_tiers).items():
            penalties.setdefault(hotkey, []).extend(reasons)
        for hotkey, reasons in self.canary_penalties(now, window_s).items():
            penalties.setdefault(hotkey, []).extend(reasons)
        for hotkey, reasons in self.auditor.penalties(now, window_s).items():
            penalties.setdefault(hotkey, []).extend(reasons)
        for hotkey, reasons in (extra_penalties or {}).items():
            penalties.setdefault(hotkey, []).extend(reasons)
        attested = self.attested_hotkeys(verdicts)
        self.record_hardware(verdicts, now, window_s)
        for hotkey, reasons in hardware_conflicts(self.hardware_sightings, now, window_s).items():
            penalties.setdefault(hotkey, []).extend(reasons)
        if self.collateral is not None and self.collateral.enabled:
            gpus = {hotkey: count for hotkey, count in self.attested_gpus(verdicts).items() if hotkey in attested}
            open_gpus = {hotkey: count for hotkey, count in self.open_tier_gpus(verdicts).items() if hotkey in attested}
            for hotkey, reasons in self.collateral.penalties(gpus, now, open_gpus).items():
                penalties.setdefault(hotkey, []).extend(reasons)
        self.record_capacity(verdicts, now, window_s)
        self._save_state()
        switch = self.switch()
        self._scored = (now, window_s, switch)
        # Capacity pay: verified GPU-time of qualified runs, split over the families the switch has on; compute_scores
        # gates and caps it, and blends it in only when the switch pays for capacity.
        gpu_seconds = self.capacity.gpu_seconds(
            now, window_s, switch.capacity_min_uptime_s, lambda profile_ids: self.served_families(profile_ids, switch)
        )
        self.last_capacity = CapacityCredit(gpu_seconds)
        return compute_scores(
            audit.entries, attested, self.profiles, switch, now, window_s,
            penalties=penalties, flags=flags, tier_rates=self.tier_policy.rates(), capacity=self.last_capacity,
        )

    def step(
        self, canary_profiles: list[str] | None = None, standard_canary_profiles: list[str] | None = None,
        window_s: float = 86400.0, plan_canary_profiles: list[str] | None = None,
        standard_plan_canary_profiles: list[str] | None = None,
    ) -> dict[str, float]:
        """One serving round. The main validator challenges, sends canaries, audits, scores and publishes its findings;
        an auditor verifies published evidence with spot challenges, applies the main validator's signed findings,
        scores, and measures how far its weights are from the main validator's."""
        extra_penalties: dict[str, list[str]] | None = None
        self._landmark_list = None  # the owner may publish a new landmark list between rounds
        if self.role == "auditor":
            if canary_profiles or standard_canary_profiles or plan_canary_profiles or standard_plan_canary_profiles:
                log.warning("auditor validators send no canaries; ignoring the canary profiles given")
            verdicts = self.published_verdicts()
            extra_penalties = self.main_validator_findings(time.time(), window_s)
        else:
            verdicts = self.check_enclaves()
        for eid, verdict in verdicts.items():
            if not verdict.ok:
                log.warning("enclave %s failed attestation: %s", eid, "; ".join(verdict.reasons))
        if self.role == "main":
            canaries = [(p, "private") for p in canary_profiles or []] + [(p, "standard") for p in standard_canary_profiles or []]
            for profile_id, privacy in canaries:
                outcome = self.run_canary(profile_id, privacy)
                level = logging.INFO if outcome.ok else logging.WARNING
                log.log(level, "%s canary %s on %s: %s", privacy, profile_id, outcome.miner_hotkey or "unknown miner", outcome.detail)
            plans = [(p, "private") for p in plan_canary_profiles or []] + [(p, "standard") for p in standard_plan_canary_profiles or []]
            for profile_id, privacy in plans:
                outcome = self.run_plan_canary(profile_id, privacy)
                level = logging.INFO if outcome.ok else logging.WARNING
                log.log(level, "%s plan canary %s on %s: %s", privacy, profile_id, outcome.miner_hotkey or "unknown miner", outcome.detail)
            self.run_audits()
        scores = self.score(verdicts, window_s, extra_penalties)
        weights = self._weights(scores)
        now = self._scored[0] if self._scored is not None else time.time()
        if self.role == "main":
            self.publish_findings(weights, now, window_s)
        elif self._main_weights is not None:
            self.last_divergence = weight_divergence(weights, self._main_weights)
            level = logging.WARNING if self.last_divergence > self.divergence_warning else logging.INFO
            log.log(level, "weights differ from the main validator's by %.1f%% of total weight", 100 * self.last_divergence)
        else:
            self.last_divergence = None
        return weights

    def _weights(self, scores: dict[str, MinerScore]) -> dict[str, float]:
        for miner in scores.values():
            log.info(
                "miner %s: score=%.4f ok=%d failed=%d %s%s",
                miner.hotkey, miner.score, miner.succeeded, miner.failed, "; ".join(miner.reasons),
                f" [flags: {'; '.join(miner.flags)}]" if miner.flags else "",
            )
        usd = self.pay is not None and self.pay.policy.usd
        for family, ready in sorted((self.last_capacity.families if self.last_capacity is not None else {}).items()):
            log.info(
                "capacity %s: %.2f verified GPUs on average over the window for a target of %d (utilization %.2f, credit scale %.3f)%s",
                family, ready.average_gpus, ready.target, ready.utilization, ready.scale,
                "" if usd else f"; {ready.blend:.3f} of the family's split paid for capacity",
            )
        if self.pay is not None and self.pay.policy.usd and self._scored is not None:
            # USD-denominated pay: the same gates, with work priced by the owner-signed rate card. PayUnavailable
            # propagates, so the caller leaves the previous weights in place (and publishes no findings).
            now, window_s, switch = self._scored
            entries = self.last_audit.entries if self.last_audit is not None else []
            try:
                return self.pay.weights(scores, entries, self.profiles, switch, now, window_s, self.last_capacity)
            finally:
                self._save_state()
        return normalize(scores)

    # ------------------------------------------------------------ state

    def _load_state(self) -> None:
        """Restores the accepted switch and canary history, so a restart cannot roll either back."""
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
        except ValueError:
            log.error("validator state file %s is unreadable; starting without history", self.state_path)
            return
        if state.get("switch"):
            stored = SignedSwitch.model_validate(state["switch"])
            if self.owner_public_key is None or stored.verify(self.owner_public_key):
                self._switch = stored
            else:
                log.warning("stored switch is not signed by the configured owner key; discarded")
        self.canary_results = [CanaryResult(**item) for item in state.get("canaries", [])]
        self.hardware_sightings = {
            token: {"kind": entry.get("kind"), "hotkeys": {h: list(v) for h, v in (entry.get("hotkeys") or {}).items()}}
            for token, entry in (state.get("hardware") or {}).items()
            if isinstance(entry, dict)
        }
        if self.collateral is not None and state.get("collateral"):
            self.collateral.load(state["collateral"])
        self.admission.load(state.get("admission") or {})
        # Verified GPU runs for capacity pay, so a restart doesn't restart every GPU's uptime.
        self.capacity.load(state.get("capacity"))
        self.enclave_tiers = {k: v for k, v in (state.get("tiers") or {}).items() if isinstance(v, str)}
        # Kept even when USD pay is off, so switching modes back and forth can't roll the accepted card back.
        self._stored_rate_card = state.get("rate_card") if isinstance(state.get("rate_card"), dict) else None
        if self.pay is not None:
            self.pay.load(self._stored_rate_card)

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        cutoff = time.time() - CANARY_RETENTION_S
        self.canary_results = [r for r in self.canary_results if r.at >= cutoff]
        state = {
            "switch": self._switch.model_dump(mode="json") if self._switch is not None else None,
            "canaries": [asdict(r) for r in self.canary_results],
            "hardware": self.hardware_sightings,
            "collateral": self.collateral.dump() if self.collateral is not None else None,
            "admission": self.admission.dump(),
            "tiers": self.enclave_tiers,
            "rate_card": (self.pay.dump() if self.pay is not None else None) or self._stored_rate_card,
            "capacity": self.capacity.dump(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_path)
