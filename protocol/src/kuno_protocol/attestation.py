"""Attestation evidence: proving a worker is the pinned image inside a real TEE.

Binding (computed inside the confidential VM for a challenge nonce N):

    gpu_nonce   = SHA-256("kuno/v1/gpu" | N | SHA-256(hpke_pk | sign_pk))
    report_data = SHA-512("kuno/v1/report" | N | SHA-256(hpke_pk | sign_pk) | SHA-256(gpu_evidence))

`report_data` is the 64-byte REPORTDATA of the TDX quote, so one quote proves:
this measured VM, on this GPU evidence, owns these keys, right now.

Two TEE backends:
  * MockTEE — development only. "Quotes" are Ed25519-signed JSON with synthetic
    measurements; the golden manifest lists which mock signing keys to trust.
  * TdxTEE  — real Intel TDX via the Linux configfs-tsm interface. Quote
    signature/TCB verification and NVIDIA GPU evidence verification plug in via
    QuoteVerifier / GpuVerifier (dcap-qvl and NVAT in production).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from .canonical import b64d, b64e, canonical_json, sha256_hex
from .crypto import verify_signature

TeeKind = Literal["mock", "tdx"]


def enclave_id_for(hpke_public_key: bytes, signing_public_key: bytes) -> str:
    return hashlib.sha256(hpke_public_key + signing_public_key).hexdigest()[:32]


def _key_binding(hpke_public_key: bytes, signing_public_key: bytes) -> bytes:
    return hashlib.sha256(hpke_public_key + signing_public_key).digest()


def gpu_nonce_for(nonce: bytes, hpke_public_key: bytes, signing_public_key: bytes) -> bytes:
    return hashlib.sha256(b"kuno/v1/gpu" + nonce + _key_binding(hpke_public_key, signing_public_key)).digest()


def report_data_for(nonce: bytes, hpke_public_key: bytes, signing_public_key: bytes, gpu_evidence: bytes | None) -> bytes:
    return hashlib.sha512(
        b"kuno/v1/report"
        + nonce
        + _key_binding(hpke_public_key, signing_public_key)
        + hashlib.sha256(gpu_evidence or b"").digest()
    ).digest()


class AttestationEvidence(BaseModel):
    tee: TeeKind
    quote: str
    gpu_evidence: str | None = None
    nonce: str
    hpke_public_key: str
    signing_public_key: str
    image_digest: str
    profiles: list[str]
    hardware: dict[str, str | int] = Field(default_factory=dict)
    created_at: float

    @property
    def enclave_id(self) -> str:
        return enclave_id_for(b64d(self.hpke_public_key), b64d(self.signing_public_key))

    def digest(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


class AllowedMeasurement(BaseModel):
    platform: TeeKind
    image_digest: str
    profiles: list[str]
    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    rtmr3: str


class GoldenManifest(BaseModel):
    """Published, signed list of measurements the network accepts."""

    version: int = 1
    issued_at: int = Field(default_factory=lambda: int(time.time()))
    allowed: list[AllowedMeasurement] = Field(default_factory=list)
    mock_quote_keys: list[str] = Field(default_factory=list)
    max_evidence_age_s: int = 3600


# ---------------------------------------------------------------- providers


class TEEProvider(Protocol):
    kind: TeeKind

    def quote(self, report_data: bytes) -> bytes: ...

    def gpu_evidence(self, gpu_nonce: bytes) -> bytes | None: ...


def mock_measurements(image_digest: str) -> dict[str, str]:
    def m(label: str) -> str:
        return hashlib.sha384(label.encode()).hexdigest()

    return {
        "mrtd": m("kuno-mock/firmware"),
        "rtmr0": m("kuno-mock/vm-shape"),
        "rtmr1": m("kuno-mock/kernel"),
        "rtmr2": m("kuno-mock/initrd"),
        "rtmr3": m(f"kuno-mock/app/{image_digest}"),
    }


class MockTEE:
    """Development stand-in for a TDX VM. Never accepted by a production manifest."""

    kind: TeeKind = "mock"

    def __init__(self, quote_key, image_digest: str):
        self._key = quote_key
        self._measurements = mock_measurements(image_digest)

    def quote(self, report_data: bytes) -> bytes:
        body = {"tee": "mock", "measurements": self._measurements, "report_data": report_data.hex()}
        signature = self._key.sign(b"kuno/v1/mock-quote\n" + canonical_json(body))
        return canonical_json({"body": body, "signature": b64e(signature)})

    def gpu_evidence(self, gpu_nonce: bytes) -> bytes | None:
        return canonical_json({"mock_gpu": "NVIDIA H200 (simulated)", "nonce": gpu_nonce.hex(), "cc_mode": "on"})


class TdxTEE:
    """Intel TDX guest. Requires Linux ≥ 6.7 with configfs-tsm inside the confidential VM."""

    kind: TeeKind = "tdx"
    TSM_ROOT = Path("/sys/kernel/config/tsm/report")

    def quote(self, report_data: bytes) -> bytes:
        if len(report_data) != 64:
            raise ValueError("TDX REPORTDATA is exactly 64 bytes")
        entry = self.TSM_ROOT / f"kuno-{uuid.uuid4().hex}"
        entry.mkdir()
        try:
            (entry / "inblob").write_bytes(report_data)
            return (entry / "outblob").read_bytes()
        finally:
            entry.rmdir()

    def gpu_evidence(self, gpu_nonce: bytes) -> bytes | None:
        raise NotImplementedError(
            "NVIDIA GPU evidence collection is not wired yet: integrate the NVIDIA attestation SDK (NVAT) "
            "inside the CVM and return the SPDM evidence for every GPU, bound to gpu_nonce."
        )


# ---------------------------------------------------------------- TDX quote parsing

_TDX_HEADER_LEN = 48
_TDX_BODY_FIELDS = [
    ("tee_tcb_svn", 16),
    ("mrseam", 48),
    ("mrsignerseam", 48),
    ("seamattributes", 8),
    ("tdattributes", 8),
    ("xfam", 8),
    ("mrtd", 48),
    ("mrconfigid", 48),
    ("mrowner", 48),
    ("mrownerconfig", 48),
    ("rtmr0", 48),
    ("rtmr1", 48),
    ("rtmr2", 48),
    ("rtmr3", 48),
    ("reportdata", 64),
]


def parse_tdx_quote(quote: bytes) -> dict[str, str]:
    """Extracts measurements from a DCAP v4 TD quote. Does NOT verify the signature."""
    body_len = sum(size for _, size in _TDX_BODY_FIELDS)
    if len(quote) < _TDX_HEADER_LEN + body_len:
        raise ValueError("quote too short for a TDX v4 quote")
    version = int.from_bytes(quote[0:2], "little")
    tee_type = int.from_bytes(quote[4:8], "little")
    if version != 4 or tee_type != 0x81:
        raise ValueError(f"not a TDX v4 quote (version={version}, tee_type={tee_type:#x})")
    fields, offset = {}, _TDX_HEADER_LEN
    for name, size in _TDX_BODY_FIELDS:
        fields[name] = quote[offset : offset + size].hex()
        offset += size
    return fields


class QuoteVerifier(Protocol):
    def verify(self, quote: bytes) -> tuple[bool, str]:
        """Checks signature chain to Intel's root and TCB status. Returns (ok, detail)."""


class GpuVerifier(Protocol):
    def verify(self, evidence: bytes, gpu_nonce: bytes) -> tuple[bool, str]: ...


# ---------------------------------------------------------------- verification


@dataclass
class Verdict:
    ok: bool
    enclave_id: str
    reasons: list[str] = field(default_factory=list)
    measurements: dict[str, str] = field(default_factory=dict)


def verify_evidence(
    evidence: AttestationEvidence,
    manifest: GoldenManifest,
    expected_nonce: bytes | None = None,
    now: float | None = None,
    quote_verifier: QuoteVerifier | None = None,
    gpu_verifier: GpuVerifier | None = None,
) -> Verdict:
    """Everything a validator, gateway or client checks before trusting an enclave key."""
    now = time.time() if now is None else now
    reasons: list[str] = []
    try:
        hpke_pk, sign_pk = b64d(evidence.hpke_public_key), b64d(evidence.signing_public_key)
        nonce, quote = bytes.fromhex(evidence.nonce), b64d(evidence.quote)
        gpu = b64d(evidence.gpu_evidence) if evidence.gpu_evidence else None
    except ValueError:
        return Verdict(False, "", ["malformed evidence encoding"])
    verdict = Verdict(False, enclave_id_for(hpke_pk, sign_pk))

    if len(hpke_pk) != 32 or len(sign_pk) != 32:
        reasons.append("keys must be 32-byte X25519 / Ed25519 public keys")
    if expected_nonce is not None and nonce != expected_nonce:
        reasons.append("nonce does not match the challenge")
    if now - evidence.created_at > manifest.max_evidence_age_s:
        reasons.append("evidence is older than the manifest allows")

    measurements, report_data = _quote_claims(evidence.tee, quote, manifest, quote_verifier, reasons)
    verdict.measurements = measurements

    expected_rd = report_data_for(nonce, hpke_pk, sign_pk, gpu).hex()
    if report_data is not None and report_data != expected_rd:
        reasons.append("REPORTDATA does not bind this nonce, these keys and this GPU evidence")

    if evidence.tee == "tdx":
        if gpu is None:
            reasons.append("GPU evidence is required on TDX workers")
        elif gpu_verifier is None:
            reasons.append("no GPU evidence verifier configured")
        else:
            ok, detail = gpu_verifier.verify(gpu, gpu_nonce_for(nonce, hpke_pk, sign_pk))
            if not ok:
                reasons.append(f"GPU evidence rejected: {detail}")

    if measurements:
        allowed = [
            a
            for a in manifest.allowed
            if a.platform == evidence.tee
            and a.image_digest == evidence.image_digest
            and all(measurements.get(k) == getattr(a, k) for k in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3"))
        ]
        if not allowed:
            reasons.append("measurements are not in the golden manifest")
        elif not set(evidence.profiles) <= set(allowed[0].profiles):
            reasons.append("image is not approved for all claimed profiles")

    verdict.reasons = reasons
    verdict.ok = not reasons
    return verdict


def _quote_claims(
    tee: str, quote: bytes, manifest: GoldenManifest, quote_verifier: QuoteVerifier | None, reasons: list[str]
) -> tuple[dict[str, str], str | None]:
    if tee == "mock":
        try:
            doc = json.loads(quote)
            body, signature = doc["body"], b64d(doc["signature"])
        except (ValueError, KeyError, TypeError):
            reasons.append("malformed mock quote")
            return {}, None
        message = b"kuno/v1/mock-quote\n" + canonical_json(body)
        if not any(verify_signature(b64d(k), signature, message) for k in manifest.mock_quote_keys):
            reasons.append("mock quote not signed by a key in the manifest")
            return {}, None
        return dict(body.get("measurements", {})), body.get("report_data")

    if tee == "tdx":
        try:
            fields = parse_tdx_quote(quote)
        except ValueError as exc:
            reasons.append(str(exc))
            return {}, None
        if quote_verifier is None:
            reasons.append("no TDX quote verifier configured")
        else:
            ok, detail = quote_verifier.verify(quote)
            if not ok:
                reasons.append(f"TDX quote rejected: {detail}")
        measurements = {k: fields[k] for k in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3")}
        return measurements, fields["reportdata"]

    reasons.append(f"unsupported TEE {tee!r}")
    return {}, None


def build_evidence(
    provider: TEEProvider,
    nonce: bytes,
    hpke_public_key: bytes,
    signing_public_key: bytes,
    image_digest: str,
    profiles: list[str],
    hardware: dict[str, str | int] | None = None,
) -> AttestationEvidence:
    """Runs inside the confidential VM."""
    gpu = provider.gpu_evidence(gpu_nonce_for(nonce, hpke_public_key, signing_public_key))
    quote = provider.quote(report_data_for(nonce, hpke_public_key, signing_public_key, gpu))
    return AttestationEvidence(
        tee=provider.kind,
        quote=b64e(quote),
        gpu_evidence=b64e(gpu) if gpu is not None else None,
        nonce=nonce.hex(),
        hpke_public_key=b64e(hpke_public_key),
        signing_public_key=b64e(signing_public_key),
        image_digest=image_digest,
        profiles=profiles,
        hardware=hardware or {},
        created_at=time.time(),
    )
