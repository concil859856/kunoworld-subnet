"""Endorsements: what a client needs to check an enclave's hardware evidence itself, whoever relays it.

A gateway or validator checks a worker's evidence online: Intel's DCAP collateral for the TDX quote, and NVIDIA's
Remote Attestation Service (NRAS) for the GPUs. A customer's browser can do neither. Intel's collateral service
doesn't answer cross-origin requests, and NRAS needs the raw GPU evidence posted to it. So a client used to check a
quote's measurements and key binding while taking its signatures on trust from the gateway that served it. A
dishonest gateway could have handed it a quote no Intel CPU produced.

Signed material stays checkable however it travels, so the gateway now relays what Intel and NVIDIA signed:

  tdx_collateral  Intel's collateral for the quote's platform: the PCK CRL, TCB info and QE identity, with their
                  issuer chains. dcap-qvl checks it against Intel's SGX root CA, which it pins, so a relay can't
                  forge it. A relay can withhold it (the check fails) or send an older copy that hasn't expired, which
                  lets a platform revoked since then pass until that copy's nextUpdate: about a month at most.
  nvidia          NRAS's answers for the GPUs and NVSwitches in the evidence (EAT tokens NRAS signed) and the JWKS
                  entries that sign them. Each entry's x5c chain must be [signing certificate, intermediate], with the
                  intermediate's public key pinned below, so a relay can't substitute a signing key of its own.

NVIDIA doesn't publish the root the intermediate chains to ("NVIDIA Attestation Service CA 001"), so the
intermediate's SubjectPublicKeyInfo is pinned instead, as SHA-256 over its DER. The value was read on 2026-09-16 from
the JWKS nras.attestation.nvidia.com serves over TLS, and matches the pin confidential-dot-ai/attestation-rs ships.
That intermediate ("NVIDIA Attestation Service GPU Intermediate 004") is valid until 2029-12-08; when NVIDIA replaces
it, clients need the new pin (`trusted_spki`).
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Literal

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from pydantic import BaseModel, Field

from .nvidia import (
    GpuEvidenceBundle,
    GpuTokenError,
    GpuVerification,
    NvidiaResult,
    _b64url,
    _bundle_problems,
    _ueid,
    check_nras_answer,
)
from .tdx import DEFAULT_ALLOWED_TCB_STATUSES, TdxQuoteResult, dcap_module, verify_tdx_quote

# SHA-256 of the DER SubjectPublicKeyInfo of "NVIDIA Attestation Service GPU Intermediate 004" (see the module docstring).
NRAS_INTERMEDIATE_SPKI_SHA256: tuple[str, ...] = ("fd32837f954e2c45db073105166dfe6985ae0480bb113fba63b091a75affe896",)
# An NRAS token's claims are checked at the time it was issued; a relay may not present one older than this.
DEFAULT_MAX_TOKEN_AGE_S = 3600
LEEWAY_S = 60


class Endorsements(BaseModel):
    """Served next to an enclave's evidence. `v` versions the layout; unknown versions are refused."""

    v: Literal[1] = 1
    tdx_collateral: dict[str, str] | None = None
    nvidia: list[NvidiaResult] = Field(default_factory=list)

    def result(self, device: str) -> NvidiaResult | None:
        return next((r for r in self.nvidia if r.device == device), None)


# ---------------------------------------------------------------- NVIDIA: tokens under a pinned intermediate


def pinned_nras_key(jwk: dict, at: float, trusted_spki: Iterable[str] = NRAS_INTERMEDIATE_SPKI_SHA256) -> ec.EllipticCurvePublicKey:
    """The P-384 key a JWKS entry names, once its certificate chain ends in a pinned intermediate valid at `at`."""
    chain = jwk.get("x5c")
    if not isinstance(chain, list) or len(chain) != 2:
        raise GpuTokenError("an NRAS signing key must carry its certificate and the intermediate that issued it (x5c of two)")
    try:
        leaf, intermediate = (x509.load_der_x509_certificate(base64.b64decode(c)) for c in chain)
    except (ValueError, TypeError) as exc:
        raise GpuTokenError("malformed certificate in an NRAS signing key") from exc
    spki = intermediate.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    if hashlib.sha256(spki).hexdigest() not in set(trusted_spki):
        raise GpuTokenError("the NRAS signing key does not chain to NVIDIA's pinned attestation intermediate")
    try:
        leaf.verify_directly_issued_by(intermediate)
    except (ValueError, TypeError, InvalidSignature) as exc:
        raise GpuTokenError("the NRAS signing certificate is not signed by the pinned intermediate") from exc
    when = datetime.fromtimestamp(at, timezone.utc)
    for name, cert in (("signing certificate", leaf), ("intermediate", intermediate)):
        if not cert.not_valid_before_utc <= when <= cert.not_valid_after_utc:
            raise GpuTokenError(f"the NRAS {name} was not valid when the token was issued")
    key = leaf.public_key()
    if not isinstance(key, ec.EllipticCurvePublicKey) or key.curve.name != "secp384r1":
        raise GpuTokenError("the NRAS signing certificate does not hold a P-384 key")
    numbers = key.public_numbers()
    if jwk.get("x") is not None and (
        _b64url(jwk["x"]) != numbers.x.to_bytes(48, "big") or _b64url(jwk.get("y", "")) != numbers.y.to_bytes(48, "big")
    ):
        raise GpuTokenError("the JWKS entry's key is not the key its certificate holds")
    return key


def verify_endorsed_token(
    token: str, keys: list[dict], now: float, *, trusted_spki: Iterable[str] = NRAS_INTERMEDIATE_SPKI_SHA256,
    max_age_s: float = DEFAULT_MAX_TOKEN_AGE_S,
) -> dict:
    """An ES384 NRAS token's claims, verified with a relayed JWKS entry under the pinned intermediate."""
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        header = json.loads(_b64url(header_b64))
        claims = json.loads(_b64url(payload_b64))
        signature = _b64url(signature_b64)
    except (ValueError, AttributeError) as exc:
        raise GpuTokenError("malformed token") from exc
    if not isinstance(header, dict) or header.get("alg") != "ES384":
        raise GpuTokenError("unexpected token algorithm")
    if not isinstance(claims, dict):
        raise GpuTokenError("token claims are not an object")
    jwk = next((k for k in keys if k.get("kid") == header.get("kid")), None)
    if jwk is None:
        raise GpuTokenError(f"no relayed signing key {header.get('kid')!r}")
    issued = claims.get("iat", claims.get("nbf"))  # EAT's issue time; nbf where a token has no iat
    if not isinstance(issued, (int, float)) or isinstance(issued, bool):
        raise GpuTokenError("token carries no issue time")
    key = pinned_nras_key(jwk, float(issued), trusted_spki)
    if len(signature) != 96:
        raise GpuTokenError("ES384 signatures are 96 bytes")
    der = encode_dss_signature(int.from_bytes(signature[:48], "big"), int.from_bytes(signature[48:], "big"))
    try:
        key.verify(der, f"{header_b64}.{payload_b64}".encode(), ec.ECDSA(hashes.SHA384()))
    except InvalidSignature as exc:
        raise GpuTokenError("token signature is invalid") from exc
    # Checked only after the signature: until then the times are the relay's word.
    if issued > now + LEEWAY_S:
        raise GpuTokenError("token is issued in the future")
    if now - issued > max_age_s:
        raise GpuTokenError("token is older than a client accepts")
    if isinstance(claims.get("exp"), (int, float)) and now > claims["exp"] + LEEWAY_S:
        raise GpuTokenError("token has expired")
    return claims


class EndorsedGpuVerifier:
    """`verify_evidence`'s GPU verifier from relayed NRAS answers: NrasGpuVerifier's claim checks, no network."""

    def __init__(self, endorsements: Endorsements, now: float | None = None,
                 trusted_spki: Iterable[str] = NRAS_INTERMEDIATE_SPKI_SHA256, max_age_s: float = DEFAULT_MAX_TOKEN_AGE_S):
        self.endorsements = endorsements
        self.now = time.time() if now is None else now
        self.trusted_spki = tuple(trusted_spki)
        self.max_age_s = max_age_s

    def verify(self, evidence: bytes, gpu_nonce: bytes) -> tuple[bool, str]:
        result = self.verify_devices(evidence, gpu_nonce)
        return result.ok, result.detail

    def _check(self, result: NvidiaResult, gpu_nonce: bytes, count: int, noun: str):
        def claims_of(token: str) -> dict:
            return verify_endorsed_token(token, result.keys, self.now, trusted_spki=self.trusted_spki, max_age_s=self.max_age_s)

        return check_nras_answer(result.answer, claims_of, gpu_nonce, count, noun)

    def verify_devices(self, evidence: bytes, gpu_nonce: bytes) -> GpuVerification:
        try:
            bundle = GpuEvidenceBundle.decode(evidence)
        except ValueError as exc:
            return GpuVerification(False, str(exc))
        refused = _bundle_problems(bundle, gpu_nonce)
        if refused:
            return GpuVerification(False, refused)
        gpu_result = self.endorsements.result("gpu")
        if gpu_result is None:
            return GpuVerification(False, "no NVIDIA attestation result was relayed for the GPUs")
        gpus, problems = self._check(gpu_result, gpu_nonce, len(bundle.gpus), "GPU")
        switches: list[dict] = []
        if gpus is not None and not problems and bundle.switches:
            switch_result = self.endorsements.result("switch")
            if switch_result is None:
                return GpuVerification(False, "no NVIDIA attestation result was relayed for the NVSwitches")
            attested, switch_problems = self._check(switch_result, gpu_nonce, len(bundle.switches), "NVSwitch")
            switches, problems = attested or [], [f"NVSwitch evidence: {p}" for p in switch_problems]
        if gpus is None or problems:
            return GpuVerification(False, "; ".join(problems))
        detail = f"{len(gpus)} GPU(s) attested by NRAS, checked under NVIDIA's pinned intermediate"
        return GpuVerification(True, detail, [_ueid(c) for c in gpus], bundle.cc, [_ueid(c) for c in switches])


# ---------------------------------------------------------------- Intel: the quote against relayed collateral


class EndorsedQuoteVerifier:
    """`verify_evidence`'s quote verifier from relayed Intel collateral: full DCAP verification, no network."""

    def __init__(self, endorsements: Endorsements, now: float | None = None,
                 allowed_statuses: tuple[str, ...] | list[str] = DEFAULT_ALLOWED_TCB_STATUSES):
        self.endorsements = endorsements
        self.now = time.time() if now is None else now
        self.allowed_statuses = tuple(allowed_statuses)

    def verify(self, quote: bytes) -> tuple[bool, str]:
        result = self.verify_quote(quote)
        return result.ok, result.detail

    def verify_quote(self, quote: bytes) -> TdxQuoteResult:
        if self.endorsements.tdx_collateral is None:
            return TdxQuoteResult(False, "no Intel collateral was relayed for this quote")
        try:
            collateral = dcap_module().QuoteCollateralV3.from_json(json.dumps(self.endorsements.tdx_collateral))
        except ValueError as exc:
            return TdxQuoteResult(False, f"malformed Intel collateral: {exc}")
        return verify_tdx_quote(quote, collateral=collateral, now=self.now, allowed_statuses=self.allowed_statuses)
