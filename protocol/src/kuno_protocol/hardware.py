"""Verified hardware identities: which physical CPU platform and GPUs produced a piece of evidence.

The point is Sybil resistance: one machine or one GPU must not pose as many miners. An
identity is only ever taken from evidence that verified cryptographically, never from
the worker's self-reported `AttestationEvidence.hardware` dictionary.

Where each identity comes from:

  cpu_platform / TDX   The PPID (Platform Provisioning ID) in the PCK certificate that signs
                       the quoting enclave's key. It is the mandatory 16-byte OCTET STRING at
                       OID 1.2.840.113741.1.13.1.1 inside the SGX extensions sequence
                       (1.2.840.113741.1.13.1) of every PCK certificate (Intel SGX PCK
                       Certificate and CRL Profile Specification, §"SGX Extensions"). The PCK
                       certificate is re-issued on TCB recovery but the PPID stays the same, so
                       it identifies the platform, not the TCB level. dcap-qvl >= 0.6 returns
                       the same value as `VerifiedReport.ppid`, taken from the PCK leaf only
                       after the chain verified to Intel's root (`verify_pck_cert_chain`).
  gpu / NVIDIA         The `ueid` (Universal Entity ID, RFC 9711 §4.2.1: globally unique and
                       permanent per manufactured device) claim of each GPU's EAT, as issued
                       in NRAS's signed detached tokens or NVIDIA's nvattest claims after the
                       device certificate chain verified to NVIDIA's root.
  mock                 A simulated platform id inside the signed mock quote body and simulated
                       GPU ueids inside mock GPU evidence (bound to that quote by REPORTDATA),
                       so development networks exercise the same registry paths.

Raw identifiers are never published. Each is turned into a token with a keyed hash under a
protocol-wide salt, so every gateway and validator derives the same token for the same
device and can dedupe on it, while feeds don't list raw PPIDs or GPU ids. The salt is
public: this hides identifiers from casual collection, not from someone who already has
a candidate serial and wants to confirm it.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

HardwareKind = Literal["cpu_platform", "gpu"]
HARDWARE_KINDS: tuple[str, ...] = ("cpu_platform", "gpu")
TOKEN_SALT = b"kuno/v1/hardware-id"
TOKEN_PREFIX = "hw1"

SOURCE_TDX_PPID = "intel-pck-ppid"
SOURCE_NVIDIA_UEID = "nvidia-ueid"
SOURCE_MOCK = "mock"


@dataclass(frozen=True)
class HardwareIdentity:
    """One verified device. `token` is what gets stored, published and compared."""

    kind: HardwareKind
    token: str
    source: str

    def public(self) -> dict[str, str]:
        return {"kind": self.kind, "token": self.token, "source": self.source}


def hardware_token(kind: str, raw: str | bytes) -> str:
    """Salted, domain-separated hash of a raw identifier; stable across verifiers of one kind."""
    if kind not in HARDWARE_KINDS:
        raise ValueError(f"unknown hardware kind {kind!r}")
    value = raw if isinstance(raw, bytes) else raw.strip().lower().encode()
    digest = hmac.new(TOKEN_SALT, kind.encode() + b"\n" + value, hashlib.sha256).hexdigest()
    return f"{TOKEN_PREFIX}:{digest[:40]}"


def identity(kind: HardwareKind, raw: str | bytes, source: str) -> HardwareIdentity:
    return HardwareIdentity(kind, hardware_token(kind, raw), source)


def tokens(identities: Iterable[HardwareIdentity], kind: str | None = None) -> set[str]:
    return {i.token for i in identities if kind is None or i.kind == kind}


def capacity_limit(gpu_count: int, gpus_per_worker: Iterable[int]) -> int:
    """How many jobs an enclave can run at once: any of its jobs may need the largest profile."""
    needed = max([max(int(g), 1) for g in gpus_per_worker] or [1])
    return max(gpu_count, 0) // needed


# ---------------------------------------------------------------- TDX: PPID from the PCK certificate

_TDX_HEADER_LEN = 48
_TD_REPORT_LEN = {2: 584, 3: 648}  # v5 body types: TDX 1.0 and TDX 1.5 TD reports
_QE_REPORT_LEN = 384
_CERT_TYPE_QE_REPORT = 6
_CERT_TYPE_PCK_CHAIN = 5
SGX_EXTENSIONS_OID = "1.2.840.113741.1.13.1"
PPID_OID = "1.2.840.113741.1.13.1.1"


def _u16(data: bytes, offset: int) -> int:
    if offset + 2 > len(data):
        raise ValueError("quote truncated")
    return int.from_bytes(data[offset : offset + 2], "little")


def _u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        raise ValueError("quote truncated")
    return int.from_bytes(data[offset : offset + 4], "little")


def pck_chain_pem(quote: bytes) -> bytes:
    """The PEM PCK certificate chain from a TDX v4/v5 quote's certification data.

    Layout (Intel TDX DCAP Quote Generation Library, quote format): header (48), TD report,
    signature data length, ECDSA signature (64), attestation key (64), certification data
    type 6 wrapping the QE report (384), its signature (64), QE authentication data, and
    certification data type 5: the PCK leaf, intermediate and root certificates as PEM.
    """
    version = _u16(quote, 0)
    offset = _TDX_HEADER_LEN
    if version == 4:
        offset += _TD_REPORT_LEN[2]
    elif version == 5:
        body_type = _u16(quote, offset)
        if body_type not in _TD_REPORT_LEN:
            raise ValueError(f"v5 quote body type {body_type} is not a TD report")
        offset += 6 + _TD_REPORT_LEN[body_type]
    else:
        raise ValueError(f"not a TDX v4/v5 quote (version={version})")
    offset += 4 + 64 + 64
    if _u16(quote, offset) != _CERT_TYPE_QE_REPORT:
        raise ValueError("quote certification data is not a QE report")
    offset += 2 + 4 + _QE_REPORT_LEN + 64
    offset += 2 + _u16(quote, offset)  # QE authentication data
    cert_type, size = _u16(quote, offset), _u32(quote, offset + 2)
    if cert_type != _CERT_TYPE_PCK_CHAIN:
        raise ValueError(f"quote carries certification data type {cert_type}, not a PCK certificate chain")
    offset += 6
    if offset + size > len(quote):
        raise ValueError("quote truncated inside the PCK certificate chain")
    return quote[offset : offset + size].rstrip(b"\x00")


def _der(data: bytes, offset: int) -> tuple[int, int, int]:
    """(tag, content start, content end) of the DER element at offset."""
    if offset + 2 > len(data):
        raise ValueError("truncated DER")
    tag, length = data[offset], data[offset + 1]
    start = offset + 2
    if length & 0x80:
        count = length & 0x7F
        if count == 0 or count > 4 or start + count > len(data):
            raise ValueError("unsupported DER length")
        length = int.from_bytes(data[start : start + count], "big")
        start += count
    if start + length > len(data):
        raise ValueError("truncated DER")
    return tag, start, start + length


def _oid(content: bytes) -> str:
    if not content:
        raise ValueError("empty OID")
    arc = min(content[0] // 40, 2)
    parts, value = [str(arc), str(content[0] - 40 * arc)], 0
    for byte in content[1:]:
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            parts.append(str(value))
            value = 0
    return ".".join(parts)


def ppid_from_sgx_extensions(extension_der: bytes) -> bytes:
    """The PPID from the DER value of a PCK certificate's SGX extensions (SEQUENCE of {OID, value})."""
    tag, start, end = _der(extension_der, 0)
    if tag != 0x30:
        raise ValueError("SGX extensions are not a SEQUENCE")
    offset = start
    while offset < end:
        tag, item_start, item_end = _der(extension_der, offset)
        if tag == 0x30:
            oid_tag, oid_start, oid_end = _der(extension_der, item_start)
            if oid_tag == 0x06 and _oid(extension_der[oid_start:oid_end]) == PPID_OID:
                value_tag, value_start, value_end = _der(extension_der, oid_end)
                if value_tag != 0x04 or value_end - value_start != 16:
                    raise ValueError("PPID is not a 16-byte OCTET STRING")
                return extension_der[value_start:value_end]
        offset = item_end
    raise ValueError("PCK certificate has no PPID")


def pck_ppid_from_quote(quote: bytes) -> bytes:
    """The PPID of the platform whose PCK certificate signed this quote. Does NOT verify the chain:
    call it only on a quote that a DCAP verifier accepted."""
    from cryptography import x509

    certificates = x509.load_pem_x509_certificates(pck_chain_pem(quote))
    for extension in certificates[0].extensions:
        if extension.oid.dotted_string == SGX_EXTENSIONS_OID:
            return ppid_from_sgx_extensions(extension.value.value)
    raise ValueError("the PCK leaf certificate has no SGX extensions")


# ---------------------------------------------------------------- mock TEE


def mock_platform_id(machine_id: str) -> str:
    return hashlib.sha256(b"kuno-mock/platform/" + machine_id.encode()).hexdigest()


def mock_gpu_ueids(machine_id: str, count: int) -> list[str]:
    return [hashlib.sha256(f"kuno-mock/gpu/{machine_id}/{index}".encode()).hexdigest() for index in range(count)]
