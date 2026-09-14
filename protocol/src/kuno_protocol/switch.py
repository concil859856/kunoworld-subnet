"""The owner-controlled model switch and request routing.

The owner signs a SwitchConfig. The gateway enforces it for routing and
validators read the same signed document to split emissions between families,
so miners are paid for the models the network actually serves.

Modes:
  h3    only MiniMax H3 profiles serve traffic
  ltx   only LTX-2.5 profiles serve traffic
  both  customer picks per request; default_family when unspecified
  auto  prefer default_family, fall back to the other family when the preferred
        one is unavailable (region license, disabled profile, no capacity)

Capacity pay (VALIDATING.md, "Capacity pay"): `capacity_share` of the serving miners' emission
pays for ready, attested confidential-tier GPUs, up to `capacity_targets` GPUs per family, once
a GPU has been verified continuously for `capacity_min_uptime_s`. At their defaults (0, none,
3600) these fields are left out of the signed bytes, so switches signed before they existed
still verify and pay nothing for capacity.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, Field, field_validator

from .canonical import b64d, b64e, canonical_json
from .crypto import verify_signature
from .profiles import FAMILY_H3, FAMILY_LTX, Mode, ModelProfile
from .regions import is_excluded

SwitchMode = Literal["h3", "ltx", "both", "auto"]

_MODE_FAMILIES: dict[str, frozenset[str]] = {
    "h3": frozenset({FAMILY_H3}),
    "ltx": frozenset({FAMILY_LTX}),
    "both": frozenset({FAMILY_H3, FAMILY_LTX}),
    "auto": frozenset({FAMILY_H3, FAMILY_LTX}),
}


DEFAULT_CAPACITY_MIN_UPTIME_S = 3600
# PLACEHOLDERS, NOT DECISIONS: the owner has not set capacity pay. Dev networks sign these (`placeholder_switch`, used by
# `kuno-devkit init`) so capacity pay runs end to end; a switch that doesn't set them pays nothing for capacity.
PLACEHOLDER_CAPACITY_SHARE = 0.25
PLACEHOLDER_CAPACITY_TARGETS = {FAMILY_H3: 8, FAMILY_LTX: 4}  # GPUs: two 4-GPU H3 workers, four 1-GPU LTX workers


class SwitchConfig(BaseModel):
    version: int = 1
    issued_at: int = Field(default_factory=lambda: int(time.time()))
    mode: SwitchMode = "auto"
    default_family: str = FAMILY_H3
    disabled_profiles: list[str] = Field(default_factory=list)
    # Set once MiniMax grants written authorization for the Excluded Territories.
    h3_authorized_everywhere: bool = False
    # Share of serving emissions per family; validators normalize over families in use.
    emission_split: dict[str, float] = Field(default_factory=lambda: {FAMILY_H3: 0.6, FAMILY_LTX: 0.4})
    # Capacity pay: the fraction of the serving miners' emission paid for ready confidential-tier GPUs (0 pays none);
    # family -> GPUs the network wants (a family without a target earns no capacity pay, and GPUs beyond it dilute
    # instead of adding emission); and the continuous verified uptime a GPU needs before its run counts.
    capacity_share: float = 0.0
    capacity_targets: dict[str, int] = Field(default_factory=dict)
    capacity_min_uptime_s: int = DEFAULT_CAPACITY_MIN_UPTIME_S

    @field_validator("capacity_share")
    @classmethod
    def _check_capacity_share(cls, value: float) -> float:
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError("capacity_share must be between 0 and 1")
        return value

    @field_validator("capacity_targets")
    @classmethod
    def _check_capacity_targets(cls, value: dict[str, int]) -> dict[str, int]:
        if any(target < 0 for target in value.values()):
            raise ValueError("a capacity target cannot be negative")
        return value

    @field_validator("capacity_min_uptime_s")
    @classmethod
    def _check_capacity_min_uptime(cls, value: int) -> int:
        if value < 0:
            raise ValueError("capacity_min_uptime_s cannot be negative")
        return value

    def family_enabled(self, family: str) -> bool:
        return family in _MODE_FAMILIES[self.mode]

    def profile_enabled(self, profile: ModelProfile) -> bool:
        return self.family_enabled(profile.family) and profile.id not in self.disabled_profiles

    def region_allows(self, profile: ModelProfile, country: str | None) -> bool:
        if self.h3_authorized_everywhere and profile.license.region_policy == "minimax-h3":
            return True
        return not is_excluded(profile.license.region_policy, country)

    def capacity_target(self, family: str) -> int:
        """GPUs this family's capacity pay is capped at; 0 when the switch sets no target."""
        return self.capacity_targets.get(family, 0)

    def signed_fields(self) -> dict:
        """What the owner signs. The capacity fields are left out at their defaults (0, none, 3600), so switches signed
        before they existed still verify."""
        fields = self.model_dump(mode="json")
        if not fields.get("capacity_share"):
            fields.pop("capacity_share", None)
        if not fields.get("capacity_targets"):
            fields.pop("capacity_targets", None)
        if fields.get("capacity_min_uptime_s") == DEFAULT_CAPACITY_MIN_UPTIME_S:
            fields.pop("capacity_min_uptime_s", None)
        return fields


class SignedSwitch(BaseModel):
    config: SwitchConfig
    signature: str | None = None

    def verify(self, owner_public_key: bytes) -> bool:
        return self.signature is not None and verify_signature(
            owner_public_key, b64d(self.signature), switch_message(self.config)
        )


def switch_message(config: SwitchConfig) -> bytes:
    return b"kuno/v1/switch\n" + canonical_json(config.signed_fields())


def sign_switch(owner_key, config: SwitchConfig) -> SignedSwitch:
    return SignedSwitch(config=config, signature=b64e(owner_key.sign(switch_message(config))))


def placeholder_switch(**changes) -> SwitchConfig:
    """The default switch with PLACEHOLDER capacity pay, for development networks. Every capacity number is a placeholder."""
    return SwitchConfig(
        **{"capacity_share": PLACEHOLDER_CAPACITY_SHARE, "capacity_targets": dict(PLACEHOLDER_CAPACITY_TARGETS), **changes}
    )


class RouteError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass
class Route:
    profile: ModelProfile
    requested_profile_id: str | None
    fallback_reason: str | None


def resolve_route(
    profiles: dict[str, ModelProfile],
    switch: SwitchConfig,
    mode: Mode,
    country: str | None,
    requested_profile_id: str | None = None,
    requested_family: str | None = None,
    has_capacity: Callable[[ModelProfile], bool] = lambda _p: True,
) -> Route:
    """Picks the profile that will serve a request, applying switch, license and capacity rules."""
    if requested_profile_id is not None:
        if requested_profile_id not in profiles:
            raise RouteError(404, "unknown_model", f"unknown model profile {requested_profile_id!r}")
        preferred = profiles[requested_profile_id]
        if mode not in preferred.modes:
            raise RouteError(422, "mode_unsupported", f"{preferred.name} does not support {mode.value}")
    else:
        family = requested_family or switch.default_family
        preferred = _first_profile(profiles, family, mode, switch, country, has_capacity) or _first_profile(
            profiles, family, mode, None, None, None
        )
        if preferred is None:
            preferred = _first_profile(profiles, None, mode, switch, country, has_capacity)
        if preferred is None:
            raise RouteError(409, "mode_unavailable", f"no model currently serves {mode.value}")

    reason = _unavailable_reason(preferred, switch, country, has_capacity)
    if reason is None:
        return Route(preferred, requested_profile_id, None)

    may_fall_back = switch.mode == "auto" or (switch.mode == "both" and reason != "capacity")
    if reason == "switched_off":
        may_fall_back = True
    if may_fall_back:
        alternative = _first_profile(profiles, None, mode, switch, country, has_capacity, exclude_family=preferred.family)
        if alternative is not None:
            return Route(alternative, requested_profile_id, reason)

    if reason == "region":
        raise RouteError(
            451,
            "region_restricted",
            f"{preferred.name} is not licensed in your region and no alternative model supports {mode.value}",
        )
    if reason == "capacity":
        raise RouteError(503, "no_capacity", f"no attested workers are serving {preferred.name} right now")
    raise RouteError(409, "model_disabled", f"{preferred.name} is currently switched off")


def _unavailable_reason(
    profile: ModelProfile, switch: SwitchConfig, country: str | None, has_capacity: Callable[[ModelProfile], bool]
) -> str | None:
    if not switch.profile_enabled(profile):
        return "switched_off"
    if not switch.region_allows(profile, country):
        return "region"
    if not has_capacity(profile):
        return "capacity"
    return None


def _first_profile(
    profiles: dict[str, ModelProfile],
    family: str | None,
    mode: Mode,
    switch: SwitchConfig | None,
    country: str | None,
    has_capacity: Callable[[ModelProfile], bool] | None,
    exclude_family: str | None = None,
) -> ModelProfile | None:
    for profile in profiles.values():
        if family is not None and profile.family != family:
            continue
        if exclude_family is not None and profile.family == exclude_family:
            continue
        if mode not in profile.modes:
            continue
        if switch is not None and _unavailable_reason(profile, switch, country, has_capacity or (lambda _p: True)):
            continue
        return profile
    return None
