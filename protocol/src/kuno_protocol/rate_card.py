"""The owner-signed rate card: what one verified video-second earns a miner, in US dollars.

Validators that run `KUNO_PAY_MODE=usd` (VALIDATING.md, "USD-denominated pay") multiply each
miner's verified billable seconds by the card's rate for the job's profile and tier. The owner
signs `"kuno/v1/rate-card\\n" + canonical_json(RateCard)` with the Ed25519 key that signs the
switch and the golden manifest, and `issued_at` must increase, exactly like the switch.

    python -m kuno_protocol.rate_card template --out rate-card.json            # every rate a placeholder
    python -m kuno_protocol.rate_card sign --key owner.key --card rate-card.json --out rate-card.signed.json

PLACEHOLDERS. The owner has not set miner prices. Every rate `placeholder_rate_card()` writes is a
stand-in derived from `PLACEHOLDER_USD_PER_VCU_SECOND`, and the card says `"placeholder": true`
(also the default, so a card nobody reviewed is never mistaken for a real one). Validators log a
placeholder card at error level every round.
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
from .profiles import ModelProfile, load_profiles
from .tiers import CONFIDENTIAL, OPEN

TIERS = (CONFIDENTIAL, OPEN)

# PLACEHOLDER, NOT A PRICE: USD per VCU-weighted verified second on the confidential tier. The owner sets real rates.
PLACEHOLDER_USD_PER_VCU_SECOND = 0.01
# PLACEHOLDER, NOT A PRICE: open-tier rate as a share of the confidential rate (mirrors KUNO_OPEN_TIER_RATE's default).
PLACEHOLDER_OPEN_TIER_SHARE = 0.5
PLACEHOLDER_NOTE = "PLACEHOLDER RATES: not set by the subnet owner; do not rely on them for real pay."


class RateCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    issued_at: int = Field(default_factory=lambda: int(time.time()))
    currency: Literal["USD"] = "USD"
    unit: Literal["verified_video_second"] = "verified_video_second"
    # True until the owner has reviewed every rate. Defaults to True so an unreviewed card is flagged.
    placeholder: bool = True
    # profile id -> tier ("confidential" | "open") -> USD per verified billable second.
    usd_per_second: dict[str, dict[str, float]]
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

    def rate(self, profile_id: str, tier: str) -> float | None:
        """USD per verified second for this profile on this tier, or None when the card doesn't price it."""
        return (self.usd_per_second.get(profile_id) or {}).get(tier)


class SignedRateCard(BaseModel):
    card: RateCard
    signature: str | None = None

    def verify(self, owner_public_key: bytes) -> bool:
        return self.signature is not None and verify_signature(owner_public_key, b64d(self.signature), rate_card_message(self.card))


def rate_card_message(card: RateCard) -> bytes:
    return b"kuno/v1/rate-card\n" + canonical_json(card.model_dump(mode="json"))


def sign_rate_card(owner_key, card: RateCard) -> SignedRateCard:
    return SignedRateCard(card=card, signature=b64e(owner_key.sign(rate_card_message(card))))


def placeholder_rate_card(profiles: dict[str, ModelProfile] | None = None, issued_at: int | None = None) -> RateCard:
    """A card pricing every profile from PLACEHOLDER constants. Every number in it is a placeholder."""
    profiles = profiles if profiles is not None else load_profiles()
    rates = {
        profile.id: {
            CONFIDENTIAL: round(PLACEHOLDER_USD_PER_VCU_SECOND * profile.vcu_per_output_second, 6),
            OPEN: round(PLACEHOLDER_USD_PER_VCU_SECOND * profile.vcu_per_output_second * PLACEHOLDER_OPEN_TIER_SHARE, 6),
        }
        for profile in profiles.values()
    }
    card = RateCard(usd_per_second=rates, placeholder=True, note=PLACEHOLDER_NOTE)
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
        print(f"Wrote {args.out} ({len(card.usd_per_second)} profile(s), issued_at {card.issued_at}){flag}")


if __name__ == "__main__":
    main()
