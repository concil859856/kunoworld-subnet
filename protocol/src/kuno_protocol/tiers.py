"""Miner tiers, and which tier may run a job in each privacy mode.

A private job is encrypted end to end and may run only on a confidential miner: Intel TDX with NVIDIA
confidential computing (or the simulated TEE on dev networks, which production refuses). A standard job
is readable by the platform and the GPU provider, and may run on any miner, including an open-tier miner
with no TEE. See PRIVACY_MODES.md.
"""

from __future__ import annotations

from typing import Literal

from .schemas import PrivacyMode

Tier = Literal["confidential", "open"]

CONFIDENTIAL: Tier = "confidential"
OPEN: Tier = "open"
PRIVATE: PrivacyMode = "private"
STANDARD: PrivacyMode = "standard"


def tier_for_tee(tee: str) -> Tier:
    """The tier an enclave's evidence places it in. Anything not a TEE kind is open, so it never sees private jobs."""
    return CONFIDENTIAL if tee in ("tdx", "mock") else OPEN


def hotkey_proof_required(tier: str, production: bool) -> bool:
    """Whether a registration must prove its miner hotkey. Always for the open tier: with no quote,
    the proof is the only thing binding the worker's keys to a miner, on dev networks too."""
    return production or tier != CONFIDENTIAL


def tier_serves(tier: str, privacy: str) -> bool:
    """Whether a miner of this tier may run a job in this privacy mode."""
    if privacy == PRIVATE:
        return tier == CONFIDENTIAL
    if privacy == STANDARD:
        return tier in (CONFIDENTIAL, OPEN)
    return False
