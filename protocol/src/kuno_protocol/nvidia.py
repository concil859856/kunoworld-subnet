"""NVIDIA GPU confidential-computing evidence: the wire format and its verifiers.

Evidence is what NVIDIA's own tooling produces inside the CVM: for each GPU, the SPDM
attestation report generated for our 32-byte `gpu_nonce` and the device certificate
chain (`nvattest collect-evidence`, or NVML directly). We carry it as

    canonical_json({"format": "kuno/v1/nvidia-gpu", "nonce": hex, "gpus": [{"arch", "evidence", "certificate"}]})

with `evidence` and `certificate` in the standard base64 NVIDIA's services expect, so the
bytes hashed into REPORTDATA are exactly the bytes a verifier forwards.

Two verifiers, both usable as the `GpuVerifier` hook of `verify_evidence`:

  * NrasGpuVerifier — NVIDIA Remote Attestation Service. Sends the evidence, verifies the
    returned EAT tokens' ES384 signatures against NRAS's published JWKS, and checks the
    claims itself rather than trusting a single boolean.
  * NvattestGpuVerifier — local verification with `nvattest attest --verifier local` from
    NVIDIA's Attestation SDK (NVAT). NVAT still fetches reference measurements (RIM) and
    certificate status (OCSP) from NVIDIA unless those services are mirrored.
"""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Callable, Literal, Protocol
from urllib.parse import urlparse

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from pydantic import BaseModel, Field, ValidationError

from .canonical import canonical_json

GPU_EVIDENCE_FORMAT = "kuno/v1/nvidia-gpu"
DEFAULT_NRAS_URL = "https://nras.attestation.nvidia.com/v4/attest/gpu"
GPU_ARCHITECTURES = ("HOPPER", "BLACKWELL")

HttpCall = Callable[[str, str, bytes | None, dict[str, str], float], tuple[int, bytes]]


class GpuEvidenceItem(BaseModel):
    arch: str
    evidence: str
    certificate: str


class GpuEvidenceBundle(BaseModel):
    format: Literal["kuno/v1/nvidia-gpu"] = GPU_EVIDENCE_FORMAT
    nonce: str
    gpus: list[GpuEvidenceItem] = Field(min_length=1)

    def encode(self) -> bytes:
        return canonical_json(self.model_dump(mode="json"))

    @classmethod
    def decode(cls, data: bytes) -> GpuEvidenceBundle:
        try:
            return cls.model_validate_json(data)
        except ValidationError as exc:
            raise ValueError(f"not {GPU_EVIDENCE_FORMAT} evidence") from exc


class GpuEvidenceCollector(Protocol):
    def collect(self, gpu_nonce: bytes) -> list[GpuEvidenceItem]:
        """Evidence for every GPU in the VM, generated for this 32-byte nonce."""


class GpuVerifierUnavailable(RuntimeError):
    pass


@dataclass
class GpuVerification:
    """A GPU verifier's full answer. `ueids` holds each attested GPU's `ueid` claim (None where a
    token carried none), taken only from claims whose signature or local verification succeeded."""

    ok: bool
    detail: str
    ueids: list[str | None] = dc_field(default_factory=list)

    @property
    def gpu_count(self) -> int:
        return len(self.ueids)


def _ueid(claims: dict) -> str | None:
    value = claims.get("ueid")
    return str(value).strip() if value not in (None, "") else None


class GpuTokenError(ValueError):
    pass


def gpu_claim_problems(claims: dict) -> list[str]:
    """What a relying party must see for one GPU. NRAS and NVAT spell some values differently."""
    problems = []
    if str(claims.get("measres", "")).lower() != "success":
        problems.append(f"runtime measurements do not match NVIDIA's reference values (measres={claims.get('measres')!r})")
    if claims.get("dbgstat") not in ("disabled", False):
        problems.append(f"GPU debug is not confirmed disabled (dbgstat={claims.get('dbgstat')!r})")
    if claims.get("secboot") is not True:
        problems.append("GPU secure boot is not confirmed")
    for name in ("x-nvidia-gpu-attestation-report-nonce-match", "x-nvidia-gpu-attestation-report-signature-verified"):
        if claims.get(name) is not True:
            problems.append(f"{name} is not true")
    return problems


# ---------------------------------------------------------------- JWT (ES384) without extra dependencies


def _b64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _jwk_public_key(jwk: dict) -> ec.EllipticCurvePublicKey:
    if jwk.get("x5c"):
        key = x509.load_der_x509_certificate(base64.b64decode(jwk["x5c"][0])).public_key()
    elif jwk.get("kty") == "EC" and jwk.get("crv") == "P-384":
        numbers = ec.EllipticCurvePublicNumbers(
            int.from_bytes(_b64url(jwk["x"]), "big"), int.from_bytes(_b64url(jwk["y"]), "big"), ec.SECP384R1()
        )
        key = numbers.public_key()
    else:
        raise GpuTokenError("JWKS key is not a P-384 EC key")
    if not isinstance(key, ec.EllipticCurvePublicKey) or key.curve.name != "secp384r1":
        raise GpuTokenError("JWKS key is not a P-384 EC key")
    return key


def verify_es384_jwt(token: str, jwks: dict, now: float, leeway_s: float = 60.0) -> dict:
    """Verifies a compact ES384 JWS against a JWKS and returns its claims."""
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        header = json.loads(_b64url(header_b64))
        claims = json.loads(_b64url(payload_b64))
        signature = _b64url(signature_b64)
    except (ValueError, AttributeError) as exc:
        raise GpuTokenError("malformed token") from exc
    if header.get("alg") != "ES384":
        raise GpuTokenError(f"unexpected token algorithm {header.get('alg')!r}")
    jwk = next((k for k in jwks.get("keys", []) if k.get("kid") == header.get("kid")), None)
    if jwk is None:
        raise GpuTokenError(f"unknown signing key {header.get('kid')!r}")
    if len(signature) != 96:
        raise GpuTokenError("ES384 signatures are 96 bytes")
    der = encode_dss_signature(int.from_bytes(signature[:48], "big"), int.from_bytes(signature[48:], "big"))
    try:
        _jwk_public_key(jwk).verify(der, f"{header_b64}.{payload_b64}".encode(), ec.ECDSA(hashes.SHA384()))
    except InvalidSignature as exc:
        raise GpuTokenError("token signature is invalid") from exc
    if not isinstance(claims, dict):
        raise GpuTokenError("token claims are not an object")
    if isinstance(claims.get("exp"), (int, float)) and now > claims["exp"] + leeway_s:
        raise GpuTokenError("token has expired")
    return claims


def _urllib_http(method: str, url: str, body: bytes | None, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _split_detached_eat(document) -> tuple[str, dict[str, str]]:
    """NRAS answers `[["JWT", overall_token], {"GPU-0": token, ...}]`."""
    if (
        isinstance(document, list)
        and len(document) == 2
        and isinstance(document[0], list)
        and len(document[0]) == 2
        and document[0][0] == "JWT"
        and isinstance(document[0][1], str)
        and isinstance(document[1], dict)
        and all(isinstance(v, str) for v in document[1].values())
    ):
        return document[0][1], document[1]
    raise GpuTokenError("unexpected NRAS response shape")


# ---------------------------------------------------------------- verifiers


class NrasGpuVerifier:
    def __init__(
        self,
        url: str = DEFAULT_NRAS_URL,
        service_key: str | None = None,
        claims_version: str = "3.0",
        timeout_s: float = 30.0,
        jwks_ttl_s: float = 3600.0,
        http: HttpCall | None = None,
        clock: Callable[[], float] = time.time,
    ):
        parsed = urlparse(url)
        self.url = url
        self.jwks_url = f"{parsed.scheme}://{parsed.netloc}/.well-known/jwks.json"
        self._service_key = service_key
        self.claims_version = claims_version
        self.timeout_s = timeout_s
        self._jwks_ttl = jwks_ttl_s
        self._http = http or _urllib_http
        self._clock = clock
        self._jwks: tuple[dict, float] | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"NrasGpuVerifier({self.url!r})"

    def _keys(self, refresh: bool = False) -> dict:
        now = self._clock()
        with self._lock:
            cached = self._jwks
        if cached is not None and not refresh and cached[1] > now:
            return cached[0]
        status, body = self._http("GET", self.jwks_url, None, {"accept": "application/json"}, self.timeout_s)
        if status != 200:
            raise GpuTokenError(f"NRAS JWKS returned HTTP {status}")
        jwks = json.loads(body)
        with self._lock:
            self._jwks = (jwks, now + self._jwks_ttl)
        return jwks

    def _claims(self, token: str) -> dict:
        try:
            return verify_es384_jwt(token, self._keys(), self._clock())
        except GpuTokenError as exc:
            if "unknown signing key" not in str(exc):
                raise
            return verify_es384_jwt(token, self._keys(refresh=True), self._clock())

    def verify(self, evidence: bytes, gpu_nonce: bytes) -> tuple[bool, str]:
        result = self.verify_devices(evidence, gpu_nonce)
        return result.ok, result.detail

    def verify_devices(self, evidence: bytes, gpu_nonce: bytes) -> GpuVerification:
        try:
            bundle = GpuEvidenceBundle.decode(evidence)
        except ValueError as exc:
            return GpuVerification(False, str(exc))
        if bundle.nonce.lower() != gpu_nonce.hex():
            return GpuVerification(False, "GPU evidence was collected for a different nonce")
        archs = {g.arch for g in bundle.gpus}
        if len(archs) != 1:
            return GpuVerification(False, "GPUs of mixed architectures must be attested separately")
        body = {
            "nonce": gpu_nonce.hex(),
            "arch": archs.pop(),
            "evidence_list": [{"evidence": g.evidence, "certificate": g.certificate} for g in bundle.gpus],
            "claims_version": self.claims_version,
        }
        headers = {"content-type": "application/json", "accept": "application/json"}
        if self._service_key:
            headers["authorization"] = f"Bearer {self._service_key}"
        try:
            status, payload = self._http("POST", self.url, json.dumps(body).encode(), headers, self.timeout_s)
            if status != 200:
                return GpuVerification(False, f"NRAS returned HTTP {status}: {payload[:200].decode('utf-8', 'replace')}")
            overall_token, detached = _split_detached_eat(json.loads(payload))
            overall = self._claims(overall_token)
            per_gpu = {name: self._claims(token) for name, token in detached.items()}
        except (OSError, ValueError) as exc:  # URLError is an OSError; GpuTokenError and JSON errors are ValueErrors
            return GpuVerification(False, f"NRAS verification failed: {exc}")

        problems = []
        if overall.get("x-nvidia-overall-att-result") is not True:
            problems.append("NRAS overall attestation result is not true")
        if str(overall.get("eat_nonce", "")).lower() != gpu_nonce.hex():
            problems.append("NRAS token is for a different nonce")
        if len(per_gpu) != len(bundle.gpus):
            problems.append(f"NRAS attested {len(per_gpu)} GPU(s) but the evidence holds {len(bundle.gpus)}")
        for name, claims in sorted(per_gpu.items()):
            problems += [f"{name}: {p}" for p in gpu_claim_problems(claims)]
        if problems:
            return GpuVerification(False, "; ".join(problems))
        models = sorted({str(c.get("hwmodel", "?")) for c in per_gpu.values()})
        ueids = [_ueid(claims) for _, claims in sorted(per_gpu.items())]
        return GpuVerification(True, f"{len(per_gpu)} GPU(s) attested by NRAS ({', '.join(models)})", ueids)


class NvattestGpuVerifier:
    def __init__(
        self,
        binary: str = "nvattest",
        verifier: Literal["local", "remote"] = "local",
        relying_party_policy: Path | None = None,
        rim_url: str | None = None,
        ocsp_url: str | None = None,
        nras_url: str | None = None,
        timeout_s: float = 120.0,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.binary, self.verifier, self.policy = binary, verifier, relying_party_policy
        self.urls = {"--rim-url": rim_url, "--ocsp-url": ocsp_url, "--nras-url": nras_url}
        self.timeout_s = timeout_s
        self._run = run

    def verify(self, evidence: bytes, gpu_nonce: bytes) -> tuple[bool, str]:
        result = self.verify_devices(evidence, gpu_nonce)
        return result.ok, result.detail

    def verify_devices(self, evidence: bytes, gpu_nonce: bytes) -> GpuVerification:
        try:
            bundle = GpuEvidenceBundle.decode(evidence)
        except ValueError as exc:
            return GpuVerification(False, str(exc))
        if bundle.nonce.lower() != gpu_nonce.hex():
            return GpuVerification(False, "GPU evidence was collected for a different nonce")
        document = {
            "evidences": [{**g.model_dump(), "nonce": gpu_nonce.hex()} for g in bundle.gpus],
            "result_code": 0,
            "result_message": "Ok",
        }
        with tempfile.TemporaryDirectory(prefix="kuno-gpu-") as tmp:
            path = Path(tmp) / "evidence.json"
            path.write_text(json.dumps(document))
            command = [
                self.binary, "attest", "--device", "gpu", "--verifier", self.verifier,
                "--gpu-evidence-source", "file", "--gpu-evidence-file", str(path),
                "--nonce", gpu_nonce.hex(), "--format", "json",
            ]
            if self.policy is not None:
                command += ["--relying-party-policy", str(self.policy)]
            for flag, value in self.urls.items():
                if value:
                    command += [flag, value]
            try:
                done = self._run(command, capture_output=True, text=True, timeout=self.timeout_s)
            except FileNotFoundError:
                return GpuVerification(False, f"{self.binary} is not installed on this verifier (NVIDIA Attestation SDK CLI)")
            except subprocess.TimeoutExpired:
                return GpuVerification(False, f"nvattest did not finish within {self.timeout_s:.0f}s")
        try:
            result = json.loads(done.stdout)
        except ValueError:
            return GpuVerification(False, f"nvattest exited {done.returncode} without JSON output")
        if result.get("result_code") != 0 or done.returncode != 0:
            return GpuVerification(False, f"nvattest: {result.get('result_message', 'attestation failed')} (code {result.get('result_code')})")
        claims = [c for c in result.get("claims", []) if isinstance(c, dict)]
        problems = []
        if len(claims) != len(bundle.gpus):
            problems.append(f"nvattest returned claims for {len(claims)} GPU(s) but the evidence holds {len(bundle.gpus)}")
        for index, claim in enumerate(claims):
            if claim.get("eat_nonce") is not None and str(claim["eat_nonce"]).lower() != gpu_nonce.hex():
                problems.append(f"GPU-{index}: claims are for a different nonce")
            problems += [f"GPU-{index}: {p}" for p in gpu_claim_problems(claim)]
        if problems:
            return GpuVerification(False, "; ".join(problems))
        return GpuVerification(True, f"{len(claims)} GPU(s) attested by nvattest ({self.verifier})", [_ueid(c) for c in claims])
