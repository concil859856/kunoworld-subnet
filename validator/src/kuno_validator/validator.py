from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from kuno_protocol.attestation import AttestationEvidence, AttestationPolicy, GoldenManifest, Verdict
from kuno_protocol.canonical import canonical_json, sha256_hex
from kuno_protocol.mp4 import Mp4Error, probe
from kuno_protocol.profiles import Mode, ModelProfile, load_profiles
from kuno_protocol.receipts import Receipt, verify_receipt
from kuno_protocol.schemas import GenerationParams, JobState, JobStatus
from kuno_protocol.switch import SignedSwitch, SwitchConfig
from kuno_protocol.tiers import OPEN, tier_for_tee
from kuno_protocol.tolerance import Calibration
from kuno_protocol.turbo import is_candidate_profile_list

from .audits import AuditOutcome, AuditPolicy, Auditor, CanaryRecord
from .canaries import pick_prompt
from .collateral import CollateralGate
from .open_tier import AdmissionTracker, TierPolicy, apply_tiers, fraud_penalties, open_tier_gpus
from .ledger import DURATION_SLACK_S, EnclaveKey, LedgerAudit, audit_ledger, duration_bounds, enclave_keys
from .scoring import MinerScore, compute_scores, hardware_conflicts, normalize
from .usd_pay import UsdPay

log = logging.getLogger("kuno.validator")

LEDGER_PAGE = 5000
# Canary outcomes older than this are pruned from the state file; it must exceed any scoring window.
CANARY_RETENTION_S = 7 * 86400.0
# How often a standard canary polls its job.
STANDARD_CANARY_POLL_S = 2.0


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
    ):
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

    def check_enclaves(self, timeout_s: float = 30.0) -> dict[str, Verdict]:
        """Challenges every active enclave with our own nonce and verifies the answer ourselves."""
        # Turbo candidates are challenged by the Turbo track against their own manifest.
        enclaves = [
            e for e in self.enclaves()
            if e["status"] == "active" and not is_candidate_profile_list(list(e.get("profiles") or []))
        ]
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
                    elif verdict.ok and verdict.tier:
                        # Only our own verdicts decide an enclave's tier (rates, admission, the fraud rule).
                        self.enclave_tiers[verdict.enclave_id] = verdict.tier
                    verdicts[enclave["enclave_id"]] = verdict
                    del pending[challenge_id]
                elif answer["status"] == "expired":
                    del pending[challenge_id]
            if pending:
                time.sleep(0.5)
        for challenge_id, (enclave, _) in pending.items():
            verdicts[enclave["enclave_id"]] = Verdict(False, enclave["enclave_id"], ["did not answer the challenge in time"])
        return verdicts

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
                log.info("standard job %s isn't replayable here (no explicit seed, inputs or options); not audited", row["job_id"])
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

    def score(self, verdicts: dict[str, Verdict], window_s: float = 86400.0) -> dict[str, MinerScore]:
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
        attested = self.attested_hotkeys(verdicts)
        self.record_hardware(verdicts, now, window_s)
        for hotkey, reasons in hardware_conflicts(self.hardware_sightings, now, window_s).items():
            penalties.setdefault(hotkey, []).extend(reasons)
        if self.collateral is not None and self.collateral.enabled:
            gpus = {hotkey: count for hotkey, count in self.attested_gpus(verdicts).items() if hotkey in attested}
            open_gpus = {hotkey: count for hotkey, count in self.open_tier_gpus(verdicts).items() if hotkey in attested}
            for hotkey, reasons in self.collateral.penalties(gpus, now, open_gpus).items():
                penalties.setdefault(hotkey, []).extend(reasons)
        self._save_state()
        switch = self.switch()
        self._scored = (now, window_s, switch)
        return compute_scores(
            audit.entries, attested, self.profiles, switch, now, window_s,
            penalties=penalties, flags=flags, tier_rates=self.tier_policy.rates(),
        )

    def step(self, canary_profiles: list[str] | None = None, standard_canary_profiles: list[str] | None = None) -> dict[str, float]:
        verdicts = self.check_enclaves()
        for eid, verdict in verdicts.items():
            if not verdict.ok:
                log.warning("enclave %s failed attestation: %s", eid, "; ".join(verdict.reasons))
        canaries = [(p, "private") for p in canary_profiles or []] + [(p, "standard") for p in standard_canary_profiles or []]
        for profile_id, privacy in canaries:
            outcome = self.run_canary(profile_id, privacy)
            level = logging.INFO if outcome.ok else logging.WARNING
            log.log(level, "%s canary %s on %s: %s", privacy, profile_id, outcome.miner_hotkey or "unknown miner", outcome.detail)
        self.run_audits()
        scores = self.score(verdicts)
        for miner in scores.values():
            log.info(
                "miner %s: score=%.4f ok=%d failed=%d %s%s",
                miner.hotkey, miner.score, miner.succeeded, miner.failed, "; ".join(miner.reasons),
                f" [flags: {'; '.join(miner.flags)}]" if miner.flags else "",
            )
        if self.pay is not None and self.pay.policy.usd and self._scored is not None:
            # USD-denominated pay: the same gates, with work priced by the owner-signed rate card. PayUnavailable
            # propagates, so the caller leaves the previous weights in place.
            now, window_s, switch = self._scored
            entries = self.last_audit.entries if self.last_audit is not None else []
            try:
                return self.pay.weights(scores, entries, self.profiles, switch, now, window_s)
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
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_path)
