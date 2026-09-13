from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from kuno_protocol.attestation import AttestationEvidence, AttestationPolicy, GoldenManifest, Verdict
from kuno_protocol.canonical import sha256_hex
from kuno_protocol.mp4 import Mp4Error, probe
from kuno_protocol.profiles import ModelProfile, load_profiles
from kuno_protocol.receipts import Receipt, verify_receipt
from kuno_protocol.switch import SignedSwitch, SwitchConfig

from .canaries import pick_prompt
from .ledger import DURATION_SLACK_S, EnclaveKey, LedgerAudit, audit_ledger, duration_bounds, enclave_keys
from .scoring import MinerScore, compute_scores, normalize

log = logging.getLogger("kuno.validator")

LEDGER_PAGE = 5000
# Canary outcomes older than this are pruned from the state file; it must exceed any scoring window.
CANARY_RETENTION_S = 7 * 86400.0


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
    ):
        if not api_key:
            raise ValueError("a validator API key is required: the gateway authenticates every validator read")
        self.gateway_url = gateway_url.rstrip("/")
        self.manifest = manifest
        # The same policy the gateway runs: real TDX and GPU verifiers, and production's stricter rules.
        self.policy = policy or AttestationPolicy()
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
        self.state_path = state_path
        self._load_state()
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
        enclaves = [e for e in self.enclaves() if e["status"] == "active"]
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
                    verdict = self.policy.verify(evidence, self.manifest, expected_nonce=nonce)
                    if verdict.enclave_id != enclave["enclave_id"]:
                        verdict.ok = False
                        verdict.reasons.append("answered with different keys than the registered enclave")
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

    # ------------------------------------------------------------ canaries

    def run_canary(self, profile_id: str) -> CanaryResult:
        from kunoworld import KunoError  # only canaries need the client SDK

        profile = self.profiles[profile_id]
        try:
            result = self.sdk.generate(
                pick_prompt(),
                model=profile_id,
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
        return self._record(
            self.check_canary_output(profile, result.job_id, result.video, result.receipt, profile.limits.min_duration_s, next(iter(profile.limits.sizes)))
        )

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
        self.enclaves()
        audit = audit_ledger(self.ledger(now - window_s), self._keys, self.profiles)
        self.last_audit = audit
        penalties = {hotkey: list(reasons) for hotkey, reasons in audit.penalties.items()}
        for hotkey, reasons in self.canary_penalties(now, window_s).items():
            penalties.setdefault(hotkey, []).extend(reasons)
        return compute_scores(
            audit.entries, self.attested_hotkeys(verdicts), self.profiles, self.switch(), now, window_s,
            penalties=penalties, flags=audit.flags,
        )

    def step(self, canary_profiles: list[str] | None = None) -> dict[str, float]:
        verdicts = self.check_enclaves()
        for eid, verdict in verdicts.items():
            if not verdict.ok:
                log.warning("enclave %s failed attestation: %s", eid, "; ".join(verdict.reasons))
        for profile_id in canary_profiles or []:
            outcome = self.run_canary(profile_id)
            level = logging.INFO if outcome.ok else logging.WARNING
            log.log(level, "canary %s on %s: %s", profile_id, outcome.miner_hotkey or "unknown miner", outcome.detail)
        scores = self.score(verdicts)
        for miner in scores.values():
            log.info(
                "miner %s: score=%.4f ok=%d failed=%d %s%s",
                miner.hotkey, miner.score, miner.succeeded, miner.failed, "; ".join(miner.reasons),
                f" [flags: {'; '.join(miner.flags)}]" if miner.flags else "",
            )
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

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        cutoff = time.time() - CANARY_RETENTION_S
        self.canary_results = [r for r in self.canary_results if r.at >= cutoff]
        state = {
            "switch": self._switch.model_dump(mode="json") if self._switch is not None else None,
            "canaries": [asdict(r) for r in self.canary_results],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_path)
