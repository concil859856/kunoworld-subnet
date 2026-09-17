"""Plan canaries (PROTOCOL.md "Plans (Director)"; research/director-design_2026-09-16.md §4.6).

The main validator sends plan jobs like any customer, Private through the sealed path and Standard through
`POST /v1/standard/plans`, from a brief set whose entries each carry must-mention terms. Plans are attested and receipted
but unverified: sampled text can't be replayed across machines, so nothing here judges a plan's quality. The checks
catch the one cheap cheat, a miner returning canned or empty plans, and a worker breaking the protocol:

  * the receipt verifies against the enclave's key, and its `plan` block describes the delivered plan;
  * the plan decrypts, parses and passes `kuno_protocol.plans.validate` for the job's params and options;
  * its stitched length is within TARGET_TOLERANCE_S of the target, or a `repairs` entry says why not (the fit's own
    notes when no shot may grow or shrink further name the target);
  * every text passes the content policy;
  * at least half the brief's must-mention terms appear in the plan;
  * the planner is one the profile allows: `<precision recipe id>:<component>` for a recipe of the profile's family and
    variant that includes the component `limits.plan.planner`, under the profile's prompt version. A development
    network whose manifest trusts the mock TEE also accepts the mock worker's canned planner.

Production validators keep a private, rotating brief set (KUNO_PLAN_CANARY_BRIEFS, a JSON list of
`{"brief", "must_mention", "target_s"}`); the list here is only a fallback.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from kuno_protocol.attestation import GoldenManifest
from kuno_protocol.canonical import sha256_hex
from kuno_protocol.content_policy import ContentPolicyViolation, check_prompt
from kuno_protocol.plans import TARGET_TOLERANCE_S, Plan, PlanError, PlanOptions, plan_context, validate
from kuno_protocol.precision import load_recipes
from kuno_protocol.profiles import ModelProfile, shot_prompt
from kuno_protocol.receipts import Receipt
from kuno_protocol.schemas import GenerationParams

# The mock worker's planner (kuno_worker.backends.mock.MOCK_PLANNER): no weights, so no precision recipe.
MOCK_PLANNER = "mock/1:canned"


@dataclass(frozen=True)
class PlanBrief:
    brief: str
    # Words or short phrases any honest plan of this brief names; at least half must appear.
    must_mention: tuple[str, ...]
    target_s: float


# Fallback briefs. Each names its must-mention terms early and plainly, so a plan that ignores the brief can't pass.
FALLBACK_BRIEFS: tuple[PlanBrief, ...] = (
    PlanBrief("A lighthouse keeper climbs the tower at dusk and lights the lamp as a storm rolls in over the sea.",
              ("lighthouse", "keeper", "lamp"), 30.0),
    PlanBrief("A baker in a small village bakery shapes bread before sunrise, then opens the shop to the first customers.",
              ("baker", "bread", "bakery"), 24.0),
    PlanBrief("A red bicycle passes a fountain and a market in the autumn streets of an old town.",
              ("bicycle", "fountain", "market"), 20.0),
    PlanBrief("A potter throws a bowl on a spinning wheel in a sunlit workshop, glazes it and takes it from the kiln.",
              ("potter", "bowl", "kiln"), 36.0),
    PlanBrief("A fox crosses a snowy forest at night, stops by a frozen stream, and watches the moon rise.",
              ("fox", "snow", "moon"), 16.0),
)


def load_briefs(path: str | Path) -> list[PlanBrief]:
    """A brief set from a JSON file: `[{"brief": "...", "must_mention": ["..."], "target_s": 30}, ...]`."""
    items = json.loads(Path(path).read_text())
    briefs = []
    for item in items:
        terms = tuple(str(term) for term in item["must_mention"])
        if not item["brief"].strip() or not terms:
            raise ValueError("every plan canary brief needs text and at least one must-mention term")
        briefs.append(PlanBrief(item["brief"], terms, float(item["target_s"])))
    if not briefs:
        raise ValueError("the plan canary brief set is empty")
    return briefs


def pick_brief(briefs: list[PlanBrief] | tuple[PlanBrief, ...] | None = None, rng: random.Random | None = None) -> PlanBrief:
    return (rng or random).choice(list(briefs or FALLBACK_BRIEFS))


def mentioned(plan: Plan, terms: tuple[str, ...]) -> list[str]:
    """The terms a plan names, matched case-insensitively at word starts (so "snow" matches "snowy") across its title,
    scene, notes, beats and shot prompts."""
    text = " ".join([plan.title, plan.scene, plan.notes, *(shot.beat for shot in plan.shots), *(shot.prompt for shot in plan.shots)]).lower()
    return [term for term in terms if re.search(r"(?<![a-z0-9])" + re.escape(term.lower()), text)]


def planner_problem(plan: Plan, profile: ModelProfile, manifest: GoldenManifest | None) -> str | None:
    """Why the plan's planner isn't one the profile allows, or None."""
    limits = profile.limits.plan
    if limits is None:
        return f"{profile.id} makes no plans"
    if plan.planner.prompt_version != limits.prompt_version:
        return f"planned under prompt {plan.planner.prompt_version}, not {limits.prompt_version}"
    if plan.planner.model == MOCK_PLANNER:
        return None if manifest is not None and manifest.trusts_mock() else "planned by the mock planner outside a development network"
    recipe_id, _, component = plan.planner.model.rpartition(":")
    recipe = load_recipes().get(recipe_id)
    if recipe is None or component != limits.planner:
        return f"planner {plan.planner.model} is not {limits.planner} of a known precision recipe"
    if recipe.family != profile.family or (profile.variant and profile.variant not in recipe.variants) or component not in recipe.include:
        return f"planner {plan.planner.model} is not a recipe {profile.id} runs"
    return None


def policy_problem(plan: Plan) -> str | None:
    texts = [plan.scene, plan.title, plan.notes]
    for shot in plan.shots:
        texts += [shot.prompt, shot_prompt(plan.scene, shot.prompt), shot.beat]
    try:
        for text in texts:
            if text.strip():
                check_prompt(text)
    except ContentPolicyViolation:
        return "a plan text breaks the content policy"
    return None


def check_plan(
    plan: Plan, plan_json: bytes, receipt: Receipt, profile: ModelProfile, params: GenerationParams, options: PlanOptions,
    brief: PlanBrief, manifest: GoldenManifest | None,
) -> str | None:
    """What is wrong with a delivered plan canary, or None. The receipt's signature and job are checked by the caller;
    everything here is signed by the enclave, so every problem is the miner's."""
    body = receipt.body
    info = body.plan
    if info is None:
        return "the receipt certifies a video, not a plan"
    if sha256_hex(plan_json) != body.content_digest:
        return "the plan does not match the receipt's content digest"
    if (info.shots, info.planner, info.prompt_version) != (len(plan.shots), plan.planner.model, plan.planner.prompt_version) or not math.isclose(
        info.duration_s, plan.duration_s, abs_tol=1e-6
    ):
        return "the receipt misreports the plan it certifies"
    try:
        validate(plan, profile, context=plan_context(profile, params, options))
    except PlanError as exc:
        return f"the plan breaks the protocol's rules: {exc}"
    if abs(plan.duration_s - plan.target_s) > TARGET_TOLERANCE_S + 1e-6 and not any("target" in note for note in plan.repairs):
        return f"the plan runs {plan.duration_s:g}s for a {plan.target_s:g}s target and its repairs don't say why"
    if (problem := policy_problem(plan)) is not None:
        return problem
    found = mentioned(plan, brief.must_mention)
    if len(found) < math.ceil(len(brief.must_mention) / 2):
        return f"the plan names {len(found)} of the brief's {len(brief.must_mention)} must-mention terms"
    return planner_problem(plan, profile, manifest)
