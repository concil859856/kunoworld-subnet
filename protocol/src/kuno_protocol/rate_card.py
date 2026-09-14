"""The owner-signed rate card: what one verified video-second earns a miner, in US dollars.

Validators that run `KUNO_PAY_MODE=usd` (VALIDATING.md, "USD-denominated pay") price each miner's verified billable
work with the card. The owner signs `"kuno/v1/rate-card\\n" + canonical_json(RateCard)` with the Ed25519 key that signs
the switch and the golden manifest, and `issued_at` must increase, exactly like the switch.

    python -m kuno_protocol.rate_card template --out rate-card.json            # every rate a placeholder
    python -m kuno_protocol.rate_card sign --key owner.key --card rate-card.json --out rate-card.signed.json

A verified job is priced one of two ways:
  by VCU      when the card sets `usd_per_vcu_second` for the job's tier: the job's VCU (`ModelProfile.vcu_for`, which
              weighs resolution, fps and duration by GPU cost) × that rate. One rate pays every profile alike.
  by profile  otherwise: billable seconds × `usd_per_second[profile][tier]`, one rate per profile whatever the resolution,
              fps or duration. Cards signed before VCU rates existed price this way.

Capacity pay (VALIDATING.md, "Capacity pay") is priced per model family in `gpu_hour_usd`: USD per credited GPU-hour of
ready confidential-tier capacity. `usd_per_vcu_second` and `gpu_hour_usd` are left out of the signed bytes and of the
file while empty, so cards signed before they existed still verify and validators that predate them still read cards
that don't set them.

PLACEHOLDERS. The owner has not set miner prices. Every rate `placeholder_rate_card()` writes is a stand-in from the
PLACEHOLDER constants below (research/research_pricing.md §3), and the card says `"placeholder": true` (also the default,
so a card nobody reviewed is never mistaken for a real one). Validators log a placeholder card at error level every round.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .canonical import b64d, b64e, canonical_json
from .crypto import signing_key_from_bytes, verify_signature
from .profiles import FAMILY_H3, FAMILY_LTX, ModelProfile, load_profiles
from .tiers import CONFIDENTIAL, OPEN

TIERS = (CONFIDENTIAL, OPEN)

# PLACEHOLDER, NOT A PRICE: USD per VCU on the confidential tier. With the GPU-cost VCU weights in profiles.json it
# reproduces research/research_pricing.md §3's recommended miner rates within about ±5% (h3 5 s ≈ $0.114/s, ltx-2.5-fast
# 1080p ≈ $0.0095/s; ltx-2.5-fast 720p pays +14%). The owner sets real rates.
PLACEHOLDER_USD_PER_VCU_SECOND = 0.0019
# PLACEHOLDER, NOT A PRICE: open-tier rate as a share of the confidential rate (mirrors KUNO_OPEN_TIER_RATE's default).
# At 0.5 only RTX 4090/5090 open miners break even at 60% utilization (research/pricing/costs.md §8.2).
PLACEHOLDER_OPEN_TIER_SHARE = 0.75
# PLACEHOLDER, NOT A PRICE: USD per credited GPU-hour of ready confidential-tier capacity, per family. Each is below the
# family's lowest owned GPU cost, so an idle GPU never profits from capacity pay alone (research/pricing/costs.md §8.3).
PLACEHOLDER_USD_PER_GPU_HOUR = {FAMILY_LTX: 0.80, FAMILY_H3: 1.50}
PLACEHOLDER_NOTE = "PLACEHOLDER RATES: not set by the subnet owner; do not rely on them for real pay."


class RateCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    issued_at: int = Field(default_factory=lambda: int(time.time()))
    currency: Literal["USD"] = "USD"
    unit: Literal["verified_video_second"] = "verified_video_second"
    # True until the owner has reviewed every rate. Defaults to True so an unreviewed card is flagged.
    placeholder: bool = True
    # profile id -> tier ("confidential" | "open") -> USD per verified billable second, for tiers without a VCU rate.
    usd_per_second: dict[str, dict[str, float]]
    # tier -> USD per VCU (ModelProfile.vcu_for). A tier listed here prices every profile by VCU and ignores
    # usd_per_second. Not written when empty: validators that predate the field refuse a card with fields they don't know.
    usd_per_vcu_second: dict[str, float] = Field(default_factory=dict, exclude_if=lambda value: not value)
    # family -> USD per credited GPU-hour of ready capacity (capacity pay). Not written when empty, for the same reason.
    gpu_hour_usd: dict[str, float] = Field(default_factory=dict, exclude_if=lambda value: not value)
    note: str = ""

    @field_validator("usd_per_second")
    @classmethod
    def _check_rates(cls, value: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        for profile_id, tiers in value.items():
            for tier, rate in tiers.items():
                if tier not in TIERS:
                    raise ValueError(f"{profile_id}: unknown tier {tier!r} (expected one of {', '.join(TIERS)})")
                if not math.isfinite(rate) or rate < 0:
                    raise ValueError(f"{profile_id}/{tier}: a rate must be a finite, non-negative number")
        return value

    @field_validator("usd_per_vcu_second")
    @classmethod
    def _check_vcu_rates(cls, value: dict[str, float]) -> dict[str, float]:
        for tier, rate in value.items():
            if tier not in TIERS:
                raise ValueError(f"usd_per_vcu_second: unknown tier {tier!r} (expected one of {', '.join(TIERS)})")
            if not math.isfinite(rate) or rate < 0:
                raise ValueError(f"usd_per_vcu_second/{tier}: a rate must be a finite, non-negative number")
        return value

    @field_validator("gpu_hour_usd")
    @classmethod
    def _check_gpu_hour_rates(cls, value: dict[str, float]) -> dict[str, float]:
        for family, rate in value.items():
            if not math.isfinite(rate) or rate < 0:
                raise ValueError(f"{family}: a GPU-hour rate must be a finite, non-negative number")
        return value

    def rate(self, profile_id: str, tier: str) -> float | None:
        """USD per verified second for this profile on this tier, or None when the card doesn't price it."""
        return (self.usd_per_second.get(profile_id) or {}).get(tier)

    def vcu_rate(self, tier: str) -> float | None:
        """USD per VCU on this tier, or None when the card doesn't price the tier by VCU."""
        return self.usd_per_vcu_second.get(tier)

    def job_usd(self, profile_id: str, tier: str, seconds: float, vcu: float) -> float | None:
        """USD for one verified job: its `vcu` × the tier's VCU rate when the card sets one, else its billable `seconds`
        × the profile's rate on the tier; None when the card prices neither."""
        vcu_rate = self.vcu_rate(tier)
        if vcu_rate is not None:
            return vcu * vcu_rate
        rate = self.rate(profile_id, tier)
        return seconds * rate if rate is not None else None

    def gpu_hour_rate(self, family: str) -> float | None:
        """USD per credited GPU-hour of capacity in this family, or None when the card doesn't price it."""
        return self.gpu_hour_usd.get(family)

    def signed_fields(self) -> dict:
        """What the owner signs. `usd_per_vcu_second` and `gpu_hour_usd` are left out when empty, so cards signed before
        they existed still verify."""
        fields = self.model_dump(mode="json")
        for name in ("usd_per_vcu_second", "gpu_hour_usd"):
            if not fields.get(name):
                fields.pop(name, None)
        return fields


class SignedRateCard(BaseModel):
    card: RateCard
    signature: str | None = None

    def verify(self, owner_public_key: bytes) -> bool:
        return self.signature is not None and verify_signature(owner_public_key, b64d(self.signature), rate_card_message(self.card))


def rate_card_message(card: RateCard) -> bytes:
    return b"kuno/v1/rate-card\n" + canonical_json(card.signed_fields())


def sign_rate_card(owner_key, card: RateCard) -> SignedRateCard:
    return SignedRateCard(card=card, signature=b64e(owner_key.sign(rate_card_message(card))))


def placeholder_rate_card(profiles: dict[str, ModelProfile] | None = None, issued_at: int | None = None) -> RateCard:
    """A card pricing every profile by VCU and every family's capacity from the PLACEHOLDER constants. Every number in it
    is a placeholder."""
    profiles = profiles if profiles is not None else load_profiles()
    vcu_rates = {
        CONFIDENTIAL: PLACEHOLDER_USD_PER_VCU_SECOND,
        OPEN: round(PLACEHOLDER_USD_PER_VCU_SECOND * PLACEHOLDER_OPEN_TIER_SHARE, 8),
    }
    families = sorted({profile.family for profile in profiles.values()})
    gpu_hours = {family: PLACEHOLDER_USD_PER_GPU_HOUR[family] for family in families if family in PLACEHOLDER_USD_PER_GPU_HOUR}
    card = RateCard(usd_per_second={}, usd_per_vcu_second=vcu_rates, gpu_hour_usd=gpu_hours, placeholder=True, note=PLACEHOLDER_NOTE)
    return card if issued_at is None else card.model_copy(update={"issued_at": issued_at})


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m kuno_protocol.rate_card", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template", help="write an unsigned card whose every rate is a placeholder")
    template.add_argument("--out", type=Path, required=True)
    sign = sub.add_parser("sign", help="sign a card with the owner key (run offline)")
    sign.add_argument("--key", type=Path, required=True, help="owner Ed25519 key file (base64url)")
    sign.add_argument("--card", type=Path, required=True, help="bare or previously signed card JSON")
    sign.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "template":
        args.out.write_text(placeholder_rate_card().model_dump_json(indent=2))
        print(f"Wrote {args.out} (placeholder rates: edit them, set placeholder to false, then sign)")
    else:
        text = args.card.read_text()
        try:
            card = SignedRateCard.model_validate_json(text).card
        except ValueError:
            card = RateCard.model_validate_json(text)
        signed = sign_rate_card(signing_key_from_bytes(b64d(args.key.read_text().strip())), card)
        args.out.write_text(signed.model_dump_json(indent=2))
        flag = " — STILL A PLACEHOLDER CARD" if card.placeholder else ""
        priced = f"{len(card.usd_per_vcu_second)} VCU tier rate(s), {len(card.usd_per_second)} profile(s)"
        print(f"Wrote {args.out} ({priced}, issued_at {card.issued_at}){flag}")


if __name__ == "__main__":
    main()
