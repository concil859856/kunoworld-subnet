"""The attestation policy a gateway or validator runs with, built from environment variables.

    KUNO_ATTESTATION            dev (default) | production
    KUNO_OWNER_PUBLIC_KEY       owner Ed25519 public key, base64url; production requires it
    KUNO_TDX_VERIFY             1 builds the real verifiers on a dev network too (always on in production)
    KUNO_PCCS_URL               Intel PCS (default) or your own PCCS
    KUNO_TDX_TCB_ALLOWED        accepted TCB statuses, comma-separated (default UpToDate)
    KUNO_TDX_REJECT_ADVISORIES  Intel-SA ids to refuse even at an accepted status
    KUNO_TDX_COLLATERAL_TTL_S   upper bound on collateral caching (default 3600)
    KUNO_GPU_VERIFIER           nras (default) | local  (local runs NVIDIA's nvattest)
    KUNO_NRAS_URL               default https://nras.attestation.nvidia.com/v4/attest/gpu
    KUNO_NRAS_SERVICE_KEY       optional NVIDIA attestation service key
    KUNO_NVATTEST_BIN           path to nvattest for the local GPU verifier
    KUNO_NVATTEST_POLICY        optional relying-party Rego policy for nvattest

Both the gateway and every validator should build their policy here, so a network
accepts the same evidence everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

from .attestation import AttestationPolicy, PolicyError
from .canonical import b64d
from .nvidia import DEFAULT_NRAS_URL, NrasGpuVerifier, NvattestGpuVerifier
from .tdx import DEFAULT_ALLOWED_TCB_STATUSES, INTEL_PCS_URL, DcapQuoteVerifier


def _list(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def build_gpu_verifier(env: dict[str, str]):
    kind = env.get("KUNO_GPU_VERIFIER", "nras")
    if kind == "nras":
        return NrasGpuVerifier(env.get("KUNO_NRAS_URL", DEFAULT_NRAS_URL), service_key=env.get("KUNO_NRAS_SERVICE_KEY") or None)
    if kind == "local":
        policy = env.get("KUNO_NVATTEST_POLICY")
        return NvattestGpuVerifier(env.get("KUNO_NVATTEST_BIN", "nvattest"), relying_party_policy=Path(policy) if policy else None)
    raise PolicyError(f"KUNO_GPU_VERIFIER must be 'nras' or 'local', not {kind!r}")


def build_quote_verifier(env: dict[str, str]) -> DcapQuoteVerifier:
    return DcapQuoteVerifier(
        pccs_url=env.get("KUNO_PCCS_URL", INTEL_PCS_URL),
        allowed_statuses=_list(env.get("KUNO_TDX_TCB_ALLOWED")) or DEFAULT_ALLOWED_TCB_STATUSES,
        rejected_advisories=_list(env.get("KUNO_TDX_REJECT_ADVISORIES")),
        collateral_ttl_s=float(env.get("KUNO_TDX_COLLATERAL_TTL_S", "3600")),
    )


def policy_from_env(env: dict[str, str] | None = None) -> AttestationPolicy:
    env = dict(os.environ if env is None else env)
    mode = env.get("KUNO_ATTESTATION", "dev")
    if mode not in ("dev", "production"):
        raise PolicyError(f"KUNO_ATTESTATION must be 'dev' or 'production', not {mode!r}")
    production = mode == "production"
    owner = env.get("KUNO_OWNER_PUBLIC_KEY")
    real = production or env.get("KUNO_TDX_VERIFY") == "1"
    gpu_verifier = build_gpu_verifier(env) if real else None
    return AttestationPolicy(
        production=production,
        quote_verifier=build_quote_verifier(env) if real else None,
        gpu_verifier=gpu_verifier,
        owner_public_key=b64d(owner) if owner else None,
    )
