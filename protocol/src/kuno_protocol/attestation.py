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
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from .canonical import b64d, b64e, canonical_json, sha256_hex
from .crypto import verify_signature
from .hardware import (
    SOURCE_MOCK,
    SOURCE_NVIDIA_UEID,
    SOURCE_TDX_PPID,
    HardwareIdentity,
    identity,
    mock_gpu_ueids,
    mock_platform_id,
    pck_ppid_from_quote,
)
from .nvidia import GpuEvidenceBundle, GpuEvidenceCollector

# "open" is an open-tier miner with no TEE: it may only run standard (non-private) jobs.
TeeKind = Literal["mock", "tdx", "open"]


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


class OpenTierImage(BaseModel):
    """A worker image the owner allows on the open tier, and the profiles it may serve there."""

    image_digest: str
    profiles: list[str]


class OpenTierPolicy(BaseModel):
    """The owner's permission for miners without a TEE (PRIVACY_MODES.md, "Miner tiers").

    Absent or disabled, every open-tier registration is refused: that is the production default.
    An open-tier worker's image digest is self-reported, so `images` states which releases the
    owner expects and lets it withdraw one; it proves nothing about what the miner really runs.
    """

    enabled: bool = False
    images: list[OpenTierImage] = Field(default_factory=list)

    def refusal(self, image_digest: str, profiles: list[str]) -> str | None:
        if not self.enabled:
            return "the manifest does not allow the open tier"
        allowed = [image for image in self.images if image.image_digest == image_digest]
        if not allowed:
            return "image is not an approved open-tier image"
        if not set(profiles) <= set(allowed[0].profiles):
            return "open-tier image is not approved for all claimed profiles"
        return None


class GoldenManifest(BaseModel):
    """Published list of measurements the network accepts; the owner signs it as a SignedManifest."""

    version: int = 1
    issued_at: int = Field(default_factory=lambda: int(time.time()))
    allowed: list[AllowedMeasurement] = Field(default_factory=list)
    mock_quote_keys: list[str] = Field(default_factory=list)
    max_evidence_age_s: int = 3600
    # Open-tier (no TEE) registrations; None refuses them all.
    open_tier: OpenTierPolicy | None = None

    def trusts_mock(self) -> bool:
        return bool(self.mock_quote_keys) or any(a.platform == "mock" for a in self.allowed)

    def signed_fields(self) -> dict:
        """What the owner signs. `open_tier` is left out when unset, so manifests signed before it existed still verify."""
        fields = self.model_dump(mode="json")
        if fields.get("open_tier") is None:
            fields.pop("open_tier", None)
        return fields


# ---------------------------------------------------------------- signed manifest


class ManifestError(ValueError):
    pass


def manifest_message(manifest: GoldenManifest) -> bytes:
    return b"kuno/v1/manifest\n" + canonical_json(manifest.signed_fields())


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


# Enough simulated GPUs for the largest profile in the catalog (gpus_per_worker).
MOCK_GPUS = 4


class MockTEE:
    """Development stand-in for a TDX VM. Never accepted by a production manifest.

    It simulates one machine: a platform id in the signed quote body (the PPID's stand-in) and
    `gpus` GPU ueids in its GPU evidence, which REPORTDATA binds to the quote. Each instance is
    a new machine unless `machine_id` (or KUNO_MOCK_MACHINE_ID) names one, so tests and dev
    networks can put two workers on the same simulated hardware.
    """

    kind: TeeKind = "mock"

    def __init__(self, quote_key, image_digest: str, machine_id: str | None = None, gpus: int = MOCK_GPUS):
        self._key = quote_key
        self._measurements = mock_measurements(image_digest)
        self.machine_id = machine_id or os.environ.get("KUNO_MOCK_MACHINE_ID") or uuid.uuid4().hex
        self.gpus = gpus

    def quote(self, report_data: bytes) -> bytes:
        body = {
            "tee": "mock",
            "measurements": self._measurements,
            "report_data": report_data.hex(),
            "platform_id": mock_platform_id(self.machine_id),
        }
        signature = self._key.sign(b"kuno/v1/mock-quote\n" + canonical_json(body))
        return canonical_json({"body": body, "signature": b64e(signature)})

    def gpu_evidence(self, gpu_nonce: bytes) -> bytes | None:
        return canonical_json(
            {
                "mock_gpu": "NVIDIA H200 (simulated)",
                "nonce": gpu_nonce.hex(),
                "cc_mode": "on",
                "gpus": [{"ueid": ueid} for ueid in mock_gpu_ueids(self.machine_id, self.gpus)],
            }
        )


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


class OpenTEE:
    """No TEE at all: an open-tier miner (PRIVACY_MODES.md).

    Its evidence carries no quote and no GPU evidence, so it proves nothing about the image,
    the hardware or who can read memory. The registration's mandatory hotkey proof binds the
    worker's keys to a miner, and step audits, collateral and admission probes do the rest.
    """

    kind: TeeKind = "open"

    def quote(self, report_data: bytes) -> bytes:
        return b""

    def gpu_evidence(self, gpu_nonce: bytes) -> bytes | None:
        return None


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
    # Verified hardware identities (kuno_protocol.hardware) and the attested GPU count. Both are
    # filled only when the verdict is ok; gpu_count stays None when no verifier counted GPUs.
    hardware: list[HardwareIdentity] = field(default_factory=list)
    gpu_count: int | None = None
    # kuno_protocol.tiers tier of the evidence ("confidential" or "open"); None when it could not be parsed.
    tier: str | None = None

    def hardware_tokens(self, kind: str | None = None) -> set[str]:
        return {h.token for h in self.hardware if kind is None or h.kind == kind}


def _seal(verdict: Verdict, reasons: list[str], hardware: list[HardwareIdentity], gpu_count: int | None) -> Verdict:
    verdict.reasons = reasons
    verdict.ok = not reasons
    # Identities from a refused verdict must not be used for anything, so they aren't kept.
    verdict.hardware, verdict.gpu_count = (hardware, gpu_count) if verdict.ok else ([], None)
    return verdict


def verify_evidence(
    evidence: AttestationEvidence,
    manifest: GoldenManifest,
    expected_nonce: bytes | None = None,
    now: float | None = None,
    quote_verifier: QuoteVerifier | None = None,
    gpu_verifier: GpuVerifier | None = None,
    allow_open: bool = False,
) -> Verdict:
    """Everything a validator, gateway or client checks before trusting an enclave key.

    Open-tier evidence (`tee="open"`) proves no TEE, so it is refused unless the caller passes
    `allow_open=True` *and* the manifest's `open_tier` policy allows the image. Gateways and
    validators checking registrations pass it; a client about to seal a private job must not.
    """
    from .tiers import tier_for_tee  # tiers imports schemas, which imports this module

    now = time.time() if now is None else now
    reasons: list[str] = []
    try:
        hpke_pk, sign_pk = b64d(evidence.hpke_public_key), b64d(evidence.signing_public_key)
        nonce, quote = bytes.fromhex(evidence.nonce), b64d(evidence.quote)
        gpu = b64d(evidence.gpu_evidence) if evidence.gpu_evidence else None
    except ValueError:
        return Verdict(False, "", ["malformed evidence encoding"])
    verdict = Verdict(False, enclave_id_for(hpke_pk, sign_pk), tier=tier_for_tee(evidence.tee))

    if len(hpke_pk) != 32 or len(sign_pk) != 32:
        reasons.append("keys must be 32-byte X25519 / Ed25519 public keys")
    if expected_nonce is not None and nonce != expected_nonce:
        reasons.append("nonce does not match the challenge")
    if now - evidence.created_at > manifest.max_evidence_age_s:
        reasons.append("evidence is older than the manifest allows")

    if evidence.tee == "open":
        if not allow_open:
            reasons.append("open-tier evidence proves no TEE; this check accepts confidential-tier evidence only")
        if quote:
            reasons.append("open-tier evidence must not carry a quote")
        if gpu is not None:
            reasons.append("open-tier evidence must not carry GPU evidence")
        refusal = (manifest.open_tier or OpenTierPolicy()).refusal(evidence.image_digest, evidence.profiles)
        if refusal:
            reasons.append(refusal)
        # Nothing about the hardware is verified, so no identity is taken from it (the dict stays self-reported).
        return _seal(verdict, reasons, [], None)

    measurements, report_data, platform = _quote_claims(evidence.tee, quote, manifest, quote_verifier, reasons)
    verdict.measurements = measurements
    hardware: list[HardwareIdentity] = [platform] if platform is not None else []

    expected_rd = report_data_for(nonce, hpke_pk, sign_pk, gpu).hex()
    bound = report_data is not None and report_data == expected_rd
    if report_data is not None and not bound:
        reasons.append("REPORTDATA does not bind this nonce, these keys and this GPU evidence")

    gpu_ueids: list[str | None] | None = None
    if evidence.tee == "tdx":
        if gpu is None:
            reasons.append("GPU evidence is required on TDX workers")
        elif gpu_verifier is None:
            reasons.append("no GPU evidence verifier configured")
        else:
            gpu_nonce = gpu_nonce_for(nonce, hpke_pk, sign_pk)
            verify_devices = getattr(gpu_verifier, "verify_devices", None)
            if callable(verify_devices):
                result = verify_devices(gpu, gpu_nonce)
                ok, detail = result.ok, result.detail
                gpu_ueids = list(result.ueids) if ok else None
            else:  # a verifier that only answers yes or no: GPUs are verified but not counted
                ok, detail = gpu_verifier.verify(gpu, gpu_nonce)
            if not ok:
                reasons.append(f"GPU evidence rejected: {detail}")
    elif evidence.tee == "mock" and gpu is not None and bound:
        gpu_ueids = _mock_gpu_ueids(gpu)

    gpu_count = None
    if gpu_ueids is not None:
        gpu_count = len(gpu_ueids)
        prefix, source = ("mock:", SOURCE_MOCK) if evidence.tee == "mock" else ("", SOURCE_NVIDIA_UEID)
        gpus = [identity("gpu", prefix + ueid, source) for ueid in gpu_ueids if ueid]
        if len({g.token for g in gpus}) != len(gpus):
            # Repeating one GPU's evidence must not count it twice.
            reasons.append("the same GPU appears more than once in the GPU evidence")
        hardware.extend(gpus)

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

    return _seal(verdict, reasons, hardware, gpu_count)


def _mock_gpu_ueids(gpu: bytes) -> list[str | None] | None:
    try:
        document = json.loads(gpu)
        gpus = document["gpus"]
    except (ValueError, KeyError, TypeError):
        return None  # simulated evidence from before identities: verified, not counted
    if not isinstance(gpus, list):
        return None
    return [str(g["ueid"]) if isinstance(g, dict) and g.get("ueid") else None for g in gpus]


def _quote_claims(
    tee: str, quote: bytes, manifest: GoldenManifest, quote_verifier: QuoteVerifier | None, reasons: list[str]
) -> tuple[dict[str, str], str | None, HardwareIdentity | None]:
    if tee == "mock":
        try:
            doc = json.loads(quote)
            body, signature = doc["body"], b64d(doc["signature"])
        except (ValueError, KeyError, TypeError):
            reasons.append("malformed mock quote")
            return {}, None, None
        message = b"kuno/v1/mock-quote\n" + canonical_json(body)
        if not any(verify_signature(b64d(k), signature, message) for k in manifest.mock_quote_keys):
            reasons.append("mock quote not signed by a key in the manifest")
            return {}, None, None
        platform_id = body.get("platform_id")
        platform = identity("cpu_platform", f"mock:{platform_id}", SOURCE_MOCK) if isinstance(platform_id, str) and platform_id else None
        return dict(body.get("measurements", {})), body.get("report_data"), platform

    if tee == "tdx":
        try:
            fields = parse_tdx_quote(quote)
        except ValueError as exc:
            reasons.append(str(exc))
            return {}, None, None
        if int(fields["tdattributes"][:2], 16) & _TD_DEBUG_BIT:
            reasons.append("TD runs in debug mode, so the host can read its memory")
        platform = None
        if quote_verifier is None:
            reasons.append("no TDX quote verifier configured")
        else:
            verify_quote = getattr(quote_verifier, "verify_quote", None)
            if callable(verify_quote):
                result = verify_quote(quote)
                ok, detail, ppid = result.ok, result.detail, getattr(result, "ppid", None)
            else:
                (ok, detail), ppid = quote_verifier.verify(quote), None
            if not ok:
                reasons.append(f"TDX quote rejected: {detail}")
            else:
                if ppid is None:
                    # The verifier accepted the PCK chain carried in this quote, so its leaf names the platform.
                    try:
                        ppid = pck_ppid_from_quote(quote)
                    except ValueError:
                        ppid = None
                if ppid:
                    platform = identity("cpu_platform", bytes(ppid), SOURCE_TDX_PPID)
        measurements = {k: fields[k] for k in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3")}
        return measurements, fields["reportdata"], platform

    reasons.append(f"unsupported TEE {tee!r}")
    return {}, None, None


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
        allow_open: bool = False,
    ) -> Verdict:
        """`allow_open` as in `verify_evidence`. Production accepts open-tier evidence only when the caller
        allows it and the owner-signed manifest's `open_tier` policy is enabled for the image."""
        verdict = verify_evidence(
            evidence, manifest, expected_nonce, now, self.quote_verifier, self.gpu_verifier, allow_open=allow_open
        )
        if self.production:
            extra = []
            if evidence.tee not in ("tdx", "open"):
                extra.append(f"{evidence.tee} evidence is not accepted in production")
            if manifest.trusts_mock():
                extra.append("the manifest trusts the simulated TEE, which production forbids")
            if verdict.ok and evidence.tee == "tdx":
                # Production dedupes miners on hardware, so evidence that names no hardware can't register.
                if not verdict.hardware_tokens("cpu_platform"):
                    extra.append("the quote verified but yielded no platform identity (PPID)")
                if verdict.gpu_count is None:
                    extra.append("the GPU verifier did not report which GPUs it attested")
                elif len(verdict.hardware_tokens("gpu")) < verdict.gpu_count or verdict.gpu_count == 0:
                    extra.append("an attested GPU carries no device identity (ueid)")
            _seal(verdict, extra + verdict.reasons, verdict.hardware, verdict.gpu_count)
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
