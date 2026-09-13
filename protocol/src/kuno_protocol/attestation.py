"""Attestation evidence: proving a worker is the pinned image inside a real TEE.

Binding (computed inside the confidential VM for a challenge nonce N):

    gpu_nonce   = SHA-256("kuno/v1/gpu" | N | SHA-256(hpke_pk | sign_pk))
    report_data = SHA-512("kuno/v1/report" | N | SHA-256(hpke_pk | sign_pk) | SHA-256(gpu_evidence))

`report_data` is the 64-byte REPORTDATA of the TDX quote, so one quote proves:
this measured VM, on this GPU evidence, owns these keys, right now.

Two TEE backends:
  * MockTEE — development only. "Quotes" are Ed25519-signed JSON with synthetic
    measurements; the golden manifest lists which mock signing keys to trust.
  * TdxTEE  — real Intel TDX via the Linux configfs-tsm interface, with NVIDIA GPU
    evidence from a pluggable collector. Quote and GPU evidence verification plug in via
    QuoteVerifier / GpuVerifier (`kuno_protocol.tdx`, `kuno_protocol.nvidia`), and an
    AttestationPolicy decides what a production network accepts.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from .canonical import b64d, b64e, canonical_json, sha256_hex
from .crypto import verify_signature
from .nvidia import GpuEvidenceBundle, GpuEvidenceCollector

TeeKind = Literal["mock", "tdx"]


class AttestationUnavailable(RuntimeError):
    """This machine cannot produce attestation evidence; the message tells the operator what to fix."""


class TdxQuoteUnavailable(AttestationUnavailable):
    pass


class GpuEvidenceUnavailable(AttestationUnavailable):
    pass


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
    """Published list of measurements the network accepts; the owner signs it as a SignedManifest."""

    version: int = 1
    issued_at: int = Field(default_factory=lambda: int(time.time()))
    allowed: list[AllowedMeasurement] = Field(default_factory=list)
    mock_quote_keys: list[str] = Field(default_factory=list)
    max_evidence_age_s: int = 3600

    def trusts_mock(self) -> bool:
        return bool(self.mock_quote_keys) or any(a.platform == "mock" for a in self.allowed)


# ---------------------------------------------------------------- signed manifest


class ManifestError(ValueError):
    pass


def manifest_message(manifest: GoldenManifest) -> bytes:
    return b"kuno/v1/manifest\n" + canonical_json(manifest.model_dump(mode="json"))


class SignedManifest(BaseModel):
    manifest: GoldenManifest
    signature: str | None = None

    def verify(self, owner_public_key: bytes) -> bool:
        return self.signature is not None and verify_signature(
            owner_public_key, b64d(self.signature), manifest_message(self.manifest)
        )


def sign_manifest(owner_key, manifest: GoldenManifest) -> SignedManifest:
    return SignedManifest(manifest=manifest, signature=b64e(owner_key.sign(manifest_message(manifest))))


def parse_manifest(text: str | bytes, owner_public_key: bytes | None = None, require_signature: bool = False) -> GoldenManifest:
    """Reads a golden manifest, either bare (development) or owner-signed.

    A signed manifest must verify whenever an owner key is configured. `require_signature`
    (production) refuses bare manifests and signed ones there is no key to check.
    """
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise ManifestError("manifest is not JSON") from exc
    try:
        if isinstance(document, dict) and "manifest" in document:
            signed = SignedManifest.model_validate(document)
            if owner_public_key is not None:
                if not signed.verify(owner_public_key):
                    raise ManifestError("manifest signature does not verify against the owner key")
            elif require_signature:
                raise ManifestError("no owner public key is configured to check the manifest signature")
            return signed.manifest
        if require_signature:
            raise ManifestError("unsigned manifest refused: production requires an owner-signed manifest")
        return GoldenManifest.model_validate(document)
    except ValidationError as exc:
        raise ManifestError(f"malformed manifest: {exc.errors()[:3]}") from exc


def load_manifest(path: str | Path, owner_public_key: bytes | None = None, require_signature: bool = False) -> GoldenManifest:
    return parse_manifest(Path(path).read_text(), owner_public_key, require_signature)


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
    """Intel TDX guest. Requires Linux ≥ 6.7 with configfs-tsm inside the confidential VM,
    and NVIDIA GPUs in confidential-computing mode passed through to it."""

    kind: TeeKind = "tdx"
    TSM_ROOT = Path("/sys/kernel/config/tsm/report")

    def __init__(self, gpu_collector: GpuEvidenceCollector | None = None, tsm_root: Path | None = None):
        self.gpu_collector = gpu_collector
        self.tsm_root = tsm_root or self.TSM_ROOT

    def quote(self, report_data: bytes) -> bytes:
        if len(report_data) != 64:
            raise ValueError("TDX REPORTDATA is exactly 64 bytes")
        if not self.tsm_root.is_dir():
            raise TdxQuoteUnavailable(
                f"{self.tsm_root} does not exist: run inside a TDX guest on Linux ≥ 6.7 with CONFIG_TSM_REPORTS "
                "and configfs mounted (mount -t configfs none /sys/kernel/config)"
            )
        entry = self.tsm_root / f"kuno-{uuid.uuid4().hex}"
        try:
            entry.mkdir()
        except PermissionError as exc:
            raise TdxQuoteUnavailable(f"no permission to request quotes under {self.tsm_root}: the worker needs write access") from exc
        try:
            (entry / "inblob").write_bytes(report_data)
            return (entry / "outblob").read_bytes()
        except OSError as exc:
            raise TdxQuoteUnavailable(
                f"the TDX quote request failed ({exc.strerror or exc}): check that the host runs the quote "
                "generation service (QGS) and exposes it to the guest"
            ) from exc
        finally:
            try:
                entry.rmdir()
            except OSError:
                pass

    def gpu_evidence(self, gpu_nonce: bytes) -> bytes | None:
        if self.gpu_collector is None:
            raise GpuEvidenceUnavailable(
                "no NVIDIA GPU evidence collector is configured: put NVIDIA's nvattest CLI in the image "
                "or install kuno-worker[nvidia]"
            )
        gpus = self.gpu_collector.collect(gpu_nonce)
        if not gpus:
            raise GpuEvidenceUnavailable("the GPU evidence collector found no GPUs: check GPU passthrough into the CVM")
        return GpuEvidenceBundle(nonce=gpu_nonce.hex(), gpus=gpus).encode()


# ---------------------------------------------------------------- TDX quote parsing

_TDX_HEADER_LEN = 48
_TDX_V5_BODY_DESCRIPTOR_LEN = 6
_TDX_V5_TD_REPORT_TYPES = (2, 3)  # TDX 1.0 and TDX 1.5 TD reports; both start with the same fields
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
_TD_DEBUG_BIT = 0x01


def parse_tdx_quote(quote: bytes) -> dict[str, str]:
    """Extracts TD report fields from a DCAP v4 or v5 TD quote. Does NOT verify the signature."""
    if len(quote) < _TDX_HEADER_LEN:
        raise ValueError("quote too short for a TDX quote")
    version = int.from_bytes(quote[0:2], "little")
    tee_type = int.from_bytes(quote[4:8], "little")
    if version not in (4, 5) or tee_type != 0x81:
        raise ValueError(f"not a TDX v4/v5 quote (version={version}, tee_type={tee_type:#x})")
    offset = _TDX_HEADER_LEN
    if version == 5:
        body_type = int.from_bytes(quote[offset : offset + 2], "little")
        if body_type not in _TDX_V5_TD_REPORT_TYPES:
            raise ValueError(f"v5 quote body type {body_type} is not a TD report")
        offset += _TDX_V5_BODY_DESCRIPTOR_LEN
    if len(quote) < offset + sum(size for _, size in _TDX_BODY_FIELDS):
        raise ValueError(f"quote too short for a TDX v{version} quote")
    fields = {}
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
        if int(fields["tdattributes"][:2], 16) & _TD_DEBUG_BIT:
            reasons.append("TD runs in debug mode, so the host can read its memory")
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


# ---------------------------------------------------------------- policy


class PolicyError(ValueError):
    pass


@dataclass
class AttestationPolicy:
    """What a gateway or validator accepts.

    Development (`production=False`) behaves exactly like bare `verify_evidence`: simulated
    quotes pass if the manifest trusts their key, and TDX evidence is rejected per request
    when a verifier is missing. Production refuses to start without both verifiers and the
    owner key, only loads owner-signed manifests that trust no simulated TEE, and rejects
    anything but TDX evidence.
    """

    production: bool = False
    quote_verifier: QuoteVerifier | None = None
    gpu_verifier: GpuVerifier | None = None
    owner_public_key: bytes | None = None

    def __post_init__(self) -> None:
        if self.production:
            missing = [
                name
                for name, value in (
                    ("a TDX quote verifier", self.quote_verifier),
                    ("a GPU evidence verifier", self.gpu_verifier),
                    ("the owner public key", self.owner_public_key),
                )
                if value is None
            ]
            if missing:
                raise PolicyError("production attestation policy needs " + ", ".join(missing))

    def check_manifest(self, manifest: GoldenManifest) -> GoldenManifest:
        if self.production and manifest.trusts_mock():
            raise ManifestError("production manifests must not trust the simulated TEE")
        return manifest

    def parse_manifest(self, text: str | bytes) -> GoldenManifest:
        return self.check_manifest(parse_manifest(text, self.owner_public_key, require_signature=self.production))

    def load_manifest(self, path: str | Path) -> GoldenManifest:
        return self.parse_manifest(Path(path).read_text())

    def verify(
        self,
        evidence: AttestationEvidence,
        manifest: GoldenManifest,
        expected_nonce: bytes | None = None,
        now: float | None = None,
    ) -> Verdict:
        verdict = verify_evidence(evidence, manifest, expected_nonce, now, self.quote_verifier, self.gpu_verifier)
        if self.production:
            extra = []
            if evidence.tee != "tdx":
                extra.append(f"{evidence.tee} evidence is not accepted in production")
            if manifest.trusts_mock():
                extra.append("the manifest trusts the simulated TEE, which production forbids")
            verdict.reasons[:0] = extra
            verdict.ok = not verdict.reasons
        return verdict


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
