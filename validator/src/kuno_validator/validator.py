from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import httpx

from kuno_protocol.attestation import AttestationEvidence, GoldenManifest, Verdict, verify_evidence
from kuno_protocol.profiles import load_profiles
from kuno_protocol.switch import SignedSwitch, SwitchConfig

from .canaries import pick_prompt
from .scoring import MinerScore, compute_scores, normalize

log = logging.getLogger("kuno.validator")


@dataclass
class CanaryResult:
    profile_id: str
    ok: bool
    detail: str


class Validator:
    def __init__(
        self,
        gateway_url: str,
        api_key: str,
        manifest: GoldenManifest,
        owner_public_key: bytes | None = None,
        transport: httpx.BaseTransport | None = None,
        country: str | None = None,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.manifest = manifest
        self.owner_public_key = owner_public_key
        self.profiles = load_profiles()
        self._http = httpx.Client(
            base_url=self.gateway_url, headers={"authorization": f"Bearer {api_key}"}, timeout=60.0, transport=transport
        )
        # Canaries for region-licensed models (H3) must originate from a licensed region.
        self._sdk_args = (api_key, country, transport)
        self._sdk = None

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

    # ------------------------------------------------------------ inputs

    def switch(self) -> SwitchConfig:
        signed = SignedSwitch.model_validate(self._http.get("/v1/switch").raise_for_status().json())
        if self.owner_public_key is not None and not signed.verify(self.owner_public_key):
            log.warning("gateway switch is not signed by the owner key; using defaults")
            return SwitchConfig()
        return signed.config

    def ledger(self, since: float) -> list[dict]:
        return self._http.get("/validator/v1/ledger", params={"since": since}).raise_for_status().json()

    # ------------------------------------------------------------ attestation

    def check_enclaves(self, timeout_s: float = 30.0) -> dict[str, Verdict]:
        """Challenges every active enclave with our own nonce and verifies the answer ourselves."""
        enclaves = [e for e in self._http.get("/validator/v1/enclaves").raise_for_status().json() if e["status"] == "active"]
        pending: dict[str, tuple[dict, bytes]] = {}
        for enclave in enclaves:
            nonce = os.urandom(32)
            response = self._http.post("/validator/v1/challenges", json={"enclave_id": enclave["enclave_id"], "nonce": nonce.hex()})
            if response.status_code == 201:
                pending[response.json()["challenge_id"]] = (enclave, nonce)

        verdicts: dict[str, Verdict] = {}
        deadline = time.time() + timeout_s
        while pending and time.time() < deadline:
            for challenge_id, (enclave, nonce) in list(pending.items()):
                answer = self._http.get(f"/validator/v1/challenges/{challenge_id}").json()
                if answer["status"] == "answered" and answer["evidence"]:
                    evidence = AttestationEvidence.model_validate(answer["evidence"])
                    verdict = verify_evidence(evidence, self.manifest, expected_nonce=nonce)
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
        self._hotkeys = {e["enclave_id"]: e["miner_hotkey"] for e in enclaves}
        return verdicts

    def attested_hotkeys(self, verdicts: dict[str, Verdict]) -> set[str]:
        hotkeys = getattr(self, "_hotkeys", {})
        return {hotkeys[eid] for eid, v in verdicts.items() if v.ok and hotkeys.get(eid)}

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
            return CanaryResult(profile_id, False, exc.code)
        if result.profile_id != profile_id:
            return CanaryResult(profile_id, False, f"routed to {result.profile_id} ({result.fallback_reason}); canary did not test {profile_id}")
        body = result.receipt.body
        if result.video[4:8] != b"ftyp":
            return CanaryResult(profile_id, False, "output is not an MP4 file")
        if abs(body.video.duration_s - profile.limits.min_duration_s) > 0.5:
            return CanaryResult(profile_id, False, "duration does not match the request")
        return CanaryResult(profile_id, True, "ok")

    # ------------------------------------------------------------ scoring

    def score(self, verdicts: dict[str, Verdict], window_s: float = 86400.0) -> dict[str, MinerScore]:
        now = time.time()
        return compute_scores(self.ledger(now - window_s), self.attested_hotkeys(verdicts), self.profiles, self.switch(), now, window_s)

    def step(self, canary_profiles: list[str] | None = None) -> dict[str, float]:
        verdicts = self.check_enclaves()
        for eid, verdict in verdicts.items():
            if not verdict.ok:
                log.warning("enclave %s failed attestation: %s", eid, "; ".join(verdict.reasons))
        for profile_id in canary_profiles or []:
            outcome = self.run_canary(profile_id)
            log.info("canary %s: %s", profile_id, outcome.detail)
        scores = self.score(verdicts)
        for miner in scores.values():
            log.info(
                "miner %s: score=%.4f ok=%d failed=%d %s",
                miner.hotkey, miner.score, miner.succeeded, miner.failed, "; ".join(miner.reasons),
            )
        return normalize(scores)
