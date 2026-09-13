"""Signed HTTP client for the gateway's miner API. The worker only ever makes
outbound connections; the VM exposes no ports."""

from __future__ import annotations

import json
import time
from urllib.parse import urlencode

import httpx

from kuno_protocol.attestation import AttestationEvidence
from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import request_signature_message
from kuno_protocol.hotkey import HotkeyProof
from kuno_protocol.receipts import Receipt


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message

    @classmethod
    def from_response(cls, response: httpx.Response) -> GatewayError:
        try:
            detail = response.json().get("detail", {})
        except ValueError:
            detail = {}
        if not isinstance(detail, dict):
            detail = {"message": str(detail)}
        return cls(response.status_code, detail.get("code", "error"), detail.get("message", response.text[:200]))


class GatewayClient:
    def __init__(self, base_url: str, signing_key, enclave_id: str, timeout: float = 60.0, transport=None):
        self._key = signing_key
        self._enclave_id = enclave_id
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def _send(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        params: dict | None = None,
        signed: bool = True,
        content_type: str = "application/json",
        timeout: float | None = None,
    ) -> httpx.Response:
        if params:
            path = f"{path}?{urlencode(params)}"
        headers = {"content-type": content_type}
        if signed:
            timestamp = str(int(time.time()))
            signature = self._key.sign(request_signature_message(method, path, timestamp, body))
            headers.update(
                {"x-kuno-enclave": self._enclave_id, "x-kuno-timestamp": timestamp, "x-kuno-signature": b64e(signature)}
            )
        response = self._http.request(method, path, content=body, headers=headers, timeout=timeout or httpx.USE_CLIENT_DEFAULT)
        if response.status_code >= 400:
            raise GatewayError.from_response(response)
        return response

    @staticmethod
    def _json(obj) -> bytes:
        return json.dumps(obj).encode()

    def nonce(self) -> bytes:
        return bytes.fromhex(self._send("GET", "/miner/v1/nonce", signed=False).json()["nonce"])

    def register(
        self,
        evidence: AttestationEvidence,
        miner_hotkey: str | None,
        capacity: int,
        hotkey_proof: HotkeyProof | None = None,
        turbo_submission: dict | None = None,
    ) -> dict:
        body = {"evidence": evidence.model_dump(mode="json"), "miner_hotkey": miner_hotkey, "capacity": capacity}
        if hotkey_proof is not None:
            body["hotkey_proof"] = hotkey_proof.model_dump(mode="json")
        if turbo_submission is not None:
            # A Turbo candidate registers against its submission's measurements and earns for its hotkey.
            return self._send("POST", "/turbo/v1/enclaves", self._json({"registration": body, "submission": turbo_submission})).json()
        return self._send("POST", "/miner/v1/enclaves", self._json(body)).json()

    def pull(self, wait: float) -> dict:
        return self._send("POST", "/miner/v1/pull", params={"wait": wait}, timeout=wait + 30).json()

    def progress(self, job_id: str, progress: float, stage: str) -> bool:
        body = self._json({"progress": round(progress, 4), "stage": stage})
        return bool(self._send("POST", f"/miner/v1/jobs/{job_id}/progress", body).json().get("canceled"))

    def download_blob(self, blob_id: str) -> bytes:
        return self._send("GET", f"/miner/v1/blobs/{blob_id}", timeout=300).content

    def upload_blob(self, job_id: str, data: bytes) -> str:
        response = self._send(
            "POST", "/miner/v1/blobs", data, params={"job_id": job_id}, content_type="application/octet-stream", timeout=600
        )
        return response.json()["blob_id"]

    def complete(self, job_id: str, blob_id: str, receipt: Receipt) -> None:
        body = self._json({"output_blob_id": blob_id, "receipt": receipt.model_dump(mode="json")})
        self._send("POST", f"/miner/v1/jobs/{job_id}/complete", body)

    def fail(self, job_id: str, code: str, message: str) -> None:
        self._send("POST", f"/miner/v1/jobs/{job_id}/fail", self._json({"code": code, "message": message[:500]}))

    def retire(self) -> dict:
        """Tell the gateway this enclave is leaving, so queued jobs are released at once."""
        return self._send("POST", "/miner/v1/retire", timeout=10).json()

    def request_certificate(self, csr_pem: str) -> dict:
        """A C2PA signing certificate for this enclave's attested key: {certificate_chain_pem, not_after, ...}.
        The gateway issues only while its verification of this enclave's attestation is fresh."""
        return self._send("POST", "/miner/v1/certificate", self._json({"csr_pem": csr_pem}), timeout=30).json()

    def answer_challenge(self, challenge_id: str, evidence: AttestationEvidence, candidate: bool = False) -> dict:
        body = self._json({"evidence": evidence.model_dump(mode="json")})
        prefix = "/turbo/v1" if candidate else "/miner/v1"
        return self._send("POST", f"{prefix}/challenges/{challenge_id}", body).json()
