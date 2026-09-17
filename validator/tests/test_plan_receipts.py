"""Plans on the validator (PROTOCOL.md "Plans (Director)"): a plan receipt has `plan` and no `video`. The ledger credits it
without a video-length check, bills no seconds and pays the flat `vcu_weights.plan`, and doubts only numbers a plan can't
have (its shot count, stitched length, and GPU-seconds above PLAN_MAX_GPU_SECONDS). `plan_failed` is not a miner fault,
no audit touches a plan, video canary and Turbo checks refuse a plan receipt instead of crashing, and the plan canary
checks (plan_canaries.py) catch canned, empty or off-brief plans."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from kuno_protocol.attestation import GoldenManifest
from kuno_protocol.canonical import canonical_json, sha256_hex
from kuno_protocol.plans import Plan, PlanOptions, encode_plan, plan_context, repair, suggested_shots
from kuno_protocol.profiles import Mode, load_profiles
from kuno_protocol.receipts import PlanInfo, Receipt, ReceiptBody, VideoInfo, sign_receipt
from kuno_protocol.schemas import GenerationParams
from kuno_protocol.switch import SwitchConfig
from kuno_validator.audits import CanaryRecord
from kuno_validator.ledger import PLAN_MAX_GPU_SECONDS, audit_ledger, enclave_keys, is_plan, is_unverified
from kuno_validator.plan_canaries import FALLBACK_BRIEFS, MOCK_PLANNER, PlanBrief, check_plan, load_briefs, mentioned, pick_brief
from kuno_validator.scoring import compute_scores, job_vcu, normalize
from kuno_validator.usd_pay import list_price_usd

from test_receipt_ledger import NOW, FakeEnclave
from test_tolerance_audits import PROFILE, NoisyToyMiner, auditor_for, gateway_job, ledger_row
from test_turbo_scoring import PROMPT, Enclave, clip, delivered, dev_caption_box, judge  # noqa: F401  (clip is a fixture)
from test_validator import FakeGateway, make_validator

PROFILES = load_profiles()
FAST = PROFILES["ltx-2.5-fast"]
PLANNER = "ltx-2.5-distilled/bf16/1:prompt_enhancer"
PLAN = GenerationParams(profile_id=FAST.id, mode=Mode.PLAN, duration_s=30, resolution="720p", aspect_ratio="16:9", fps=24)
CLIP = GenerationParams(profile_id=FAST.id, mode=Mode.TEXT_TO_VIDEO, duration_s=5, resolution="720p", aspect_ratio="16:9", fps=24)
BRIEF = FALLBACK_BRIEFS[0]  # the lighthouse keeper


def digest(params: GenerationParams) -> str:
    return sha256_hex(canonical_json(params.model_dump(mode="json")))


def plan_info(**overrides) -> PlanInfo:
    return PlanInfo(**{"shots": 5, "duration_s": 30.375, "planner": PLANNER, "prompt_version": "plan/1", "output_tokens": 451, **overrides})


def entry(enclave: FakeEnclave, params: GenerationParams = PLAN, *, gpu_seconds: float = 14.0, info: PlanInfo | None = None,
          video: VideoInfo | None = None, with_params: bool = True, **fields) -> dict:
    """A succeeded ledger row whose receipt certifies a plan (or `video`, when given)."""
    enclave.jobs += 1
    job_id = f"{enclave.enclave_id[:8]}-plan-{enclave.jobs}"
    body = ReceiptBody(
        job_id=job_id, enclave_id=enclave.enclave_id, profile_id=FAST.id, image_digest="sha256:img", params_digest=digest(params),
        input_digest="0" * 64, output_digest="1" * 64, output_bytes=4386, content_digest=sha256_hex(job_id.encode()),
        attestation_digest="2" * 64, started_at=NOW - 80, finished_at=NOW - 60, gpu_seconds=gpu_seconds, miner_hotkey=enclave.hotkey,
        video=video, plan=None if video is not None else (info or plan_info()),
    )
    row = {
        "job_id": job_id, "enclave_id": enclave.enclave_id, "miner_hotkey": enclave.hotkey, "profile_id": FAST.id, "status": "succeeded",
        "error_code": None, "duration_s": params.duration_s, "resolution": params.resolution, "finished_at": NOW - 60,
        "receipt": sign_receipt(enclave.key, body).model_dump(mode="json"), **fields,
    }
    if with_params:
        row["params"] = params.model_dump(mode="json")
    return row


# ---------------------------------------------------------------- ledger and pay


def test_a_plan_is_credited_without_a_video_bills_no_seconds_and_pays_the_flat_plan_vcu():
    a, b = FakeEnclave("A"), FakeEnclave("B")
    rows = [entry(a), entry(b, CLIP, video=VideoInfo(duration_s=5, width=1280, height=704, fps=24, frames=121, audio=True))]
    result = audit_ledger(rows, enclave_keys([a.public(), b.public()]), PROFILES)
    assert result.dropped_total == 0 and not result.flags
    entries = {e["miner_hotkey"]: e for e in result.entries}
    # The 30 s target is no output: nothing billed per second, nothing checked against a video.
    assert (entries["A"]["credit"], entries["A"]["billable_s"], entries["A"]["plan"]) == (True, 0.0, True)
    assert job_vcu(FAST, entries["A"], entries["A"]["billable_s"]) == FAST.vcu_weights.plan == 27
    assert "plan" not in entries["B"]
    weights = normalize(compute_scores(result.entries, {"A", "B"}, PROFILES, SwitchConfig(), NOW, min_samples=1))
    assert weights["A"] / weights["B"] == pytest.approx(27 / FAST.vcu_at("720p", 24, 5))
    # The revenue fallback for rows without billable_usd prices a plan flat too.
    assert list_price_usd(entries["A"], FAST) == 0.1 and list_price_usd({**entries["A"], "privacy": "standard"}, FAST) == 0.08


@pytest.mark.parametrize(
    "fields, words",
    [
        (dict(gpu_seconds=0.0), "GPU-seconds"),
        (dict(gpu_seconds=PLAN_MAX_GPU_SECONDS + 1), "GPU-seconds"),
        (dict(gpu_seconds=-1.0), "GPU-seconds"),
        (dict(info=plan_info(shots=1)), "plan of 1 shots"),
        (dict(info=plan_info(shots=13)), "plan of 13 shots"),
        (dict(info=plan_info(duration_s=121.0)), "121s plan"),
    ],
)
def test_a_plan_receipt_with_numbers_no_plan_can_have_is_flagged_and_not_credited(fields, words):
    a = FakeEnclave("A")
    result = audit_ledger([entry(a, **fields)], enclave_keys([a.public()]), PROFILES)
    [audited] = result.entries
    assert audited["credit"] is False and any(words in flag for flag in result.flags["A"]), result.flags
    # The cap itself is credited.
    assert audit_ledger([entry(a, gpu_seconds=PLAN_MAX_GPU_SECONDS)], enclave_keys([a.public()]), PROFILES).entries[0]["credit"] is True


def test_a_receipt_whose_output_contradicts_the_signed_mode_is_dropped():
    a = FakeEnclave("A")
    video = VideoInfo(duration_s=30, width=1280, height=704, fps=24, frames=720, audio=True)
    rows = [entry(a, PLAN, video=video), entry(a, CLIP)]
    result = audit_ledger(rows, enclave_keys([a.public()]), PROFILES)
    assert result.entries == [] and result.dropped["receipt describes a different output than the job's mode"] == 2


def test_a_plan_row_without_params_still_pays_the_flat_weight():
    a = FakeEnclave("A")
    [audited] = audit_ledger([entry(a, with_params=False)], enclave_keys([a.public()]), PROFILES).entries
    assert audited["credit"] and audited["plan"] and job_vcu(FAST, audited, audited["billable_s"]) == 27


def test_plan_failed_is_not_a_miner_fault_and_a_plan_doesnt_show_a_miner_can_render_the_family():
    a = FakeEnclave("A")
    failed = {"job_id": "f1", "enclave_id": a.enclave_id, "miner_hotkey": "A", "profile_id": FAST.id, "status": "failed",
              "error_code": "plan_failed", "finished_at": NOW - 30, "params": PLAN.model_dump(mode="json")}
    rows = [entry(a, tier="confidential"), failed]
    result = audit_ledger(rows, enclave_keys([a.public()]), PROFILES)
    scores = compute_scores(result.entries, {"A"}, PROFILES, SwitchConfig(), NOW, min_samples=1)
    assert (scores["A"].succeeded, scores["A"].failed, scores["A"].served) == (1, 0, set())
    assert scores["A"].work == {FAST.family: 27}


# ---------------------------------------------------------------- audits, canaries and Turbo


def plan_record(miner: NoisyToyMiner, job_id: str, *, source: str = "standard", tier: str | None = "open") -> CanaryRecord:
    params = PLAN.model_copy(update={"profile_id": PROFILE.id}) if PROFILE.id != FAST.id else PLAN
    body = ReceiptBody(
        job_id=job_id, enclave_id=miner.enclave_id, profile_id=PROFILE.id, image_digest="sha256:img", params_digest=digest(params),
        input_digest="0" * 64, output_digest="1" * 64, output_bytes=10, content_digest=sha256_hex(job_id.encode()),
        attestation_digest="2" * 64, started_at=NOW - 60, finished_at=NOW - 10, gpu_seconds=12.0, plan=plan_info(),
        miner_hotkey=miner.enclave_key().miner_hotkey,
    )
    return CanaryRecord(job_id, PROFILE.id, params.model_dump(mode="json"), BRIEF.brief, 21, sign_receipt(miner.key, body).model_dump(mode="json"),
                        source=source, tier=tier)


def test_plans_are_never_sampled_selected_or_replayed_even_when_commitments_are_required():
    assert is_plan(PLAN) and is_plan(PLAN.model_dump(mode="json")) and is_unverified(PLAN) and not is_plan(CLIP) and not is_unverified(CLIP)
    miner = NoisyToyMiner(commit=False)
    auditor = auditor_for(miner, open_tier_rate=1.0, standard_rate=1.0, require_commitment=True)
    plan, clip = plan_record(miner, "job-plan"), miner.run("job-clip", source="standard")
    assert [row["job_id"] for row in auditor.sample_standard([ledger_row(plan), ledger_row(clip)], {}, NOW)] == ["job-clip"]
    assert auditor.request(plan) is None
    canary = plan_record(miner, "canary-plan", source="canary", tier="confidential")
    assert not auditor.auditable(canary) and not auditor.should_audit(canary) and auditor.select([canary]) == []
    assert auditor.standard_record(ledger_row(plan, tier="open"), gateway_job(plan, plan={"v": 1})) is None
    assert [o.job_id for o in auditor.outcomes] == []


def test_a_video_canary_answered_with_a_plan_receipt_fails_the_miner():
    honest = FakeEnclave("A")
    body = ReceiptBody(
        job_id="canary-1", enclave_id=honest.enclave_id, profile_id=FAST.id, image_digest="sha256:img", params_digest="0" * 64,
        input_digest="0" * 64, output_digest="1" * 64, output_bytes=10, content_digest=sha256_hex(b"x"), attestation_digest="2" * 64,
        started_at=NOW, finished_at=NOW + 5, gpu_seconds=5.0, plan=plan_info(), miner_hotkey="A",
    )
    validator = make_validator(FakeGateway(enclaves=[honest]))
    result = validator.check_canary_output(FAST, "canary-1", b"x", sign_receipt(honest.key, body), 2.0, "720p")
    assert not result.ok and result.attributable and result.miner_hotkey == "A" and "plan receipt" in result.detail


def test_a_turbo_benchmark_answered_with_a_plan_receipt_is_fraud(clip):
    enclave = Enclave("5Miner")
    video = clip + dev_caption_box(PROMPT)
    outcome = replace(delivered(enclave, video), receipt=enclave.receipt("job-1", video, video=None, plan=plan_info()))
    verdict = judge(enclave, outcome)
    assert verdict.status == "fraud" and "plan" in verdict.detail


# ---------------------------------------------------------------- plan canary checks


def written(params: GenerationParams = PLAN, options: PlanOptions | None = None, *, prompt: str | None = None,
            planner: str = PLANNER) -> tuple[Plan, bytes]:
    """A plan as a worker delivers one for BRIEF: the suggested shots, each naming the brief, repaired and fitted."""
    context = plan_context(FAST, params, options)
    count, length = suggested_shots(context)
    text = prompt or f"Wide shot; {BRIEF.brief} The camera pushes in."
    shots = [{"beat": f"Beat {i + 1}", "prompt": text, "duration_s": length, "join": "fresh" if i == 0 else "cut"} for i in range(count)]
    reply = json.dumps({"title": "A plan", "scene": "A rocky headland at dusk.", "shots": shots})
    plan = repair(reply, context, planner=planner, brief=BRIEF.brief).deliverable()
    return plan, encode_plan(plan)


def receipt_for(plan: Plan, plan_json: bytes, params: GenerationParams = PLAN, overrides: dict | None = None) -> Receipt:
    enclave = FakeEnclave("A")
    info = PlanInfo(shots=len(plan.shots), duration_s=plan.duration_s, planner=plan.planner.model,
                    prompt_version=plan.planner.prompt_version, output_tokens=400)
    fields = dict(
        job_id="canary-plan", enclave_id=enclave.enclave_id, profile_id=FAST.id, image_digest="sha256:img", params_digest=digest(params),
        input_digest="0" * 64, output_digest="1" * 64, output_bytes=4386, content_digest=sha256_hex(plan_json), attestation_digest="2" * 64,
        started_at=NOW, finished_at=NOW + 10, gpu_seconds=10.0, plan=info,
    )
    fields.update(overrides or {})
    return sign_receipt(enclave.key, ReceiptBody(**fields))


def verdict(plan: Plan, plan_json: bytes | None = None, receipt: Receipt | None = None, *, params: GenerationParams = PLAN,
            options: PlanOptions | None = None, brief: PlanBrief = BRIEF, manifest: GoldenManifest | None = None) -> str | None:
    plan_json = encode_plan(plan) if plan_json is None else plan_json
    return check_plan(plan, plan_json, receipt or receipt_for(plan, plan_json, params), FAST, params, options or PlanOptions(), brief,
                      manifest or GoldenManifest())


def test_an_honest_plan_canary_passes():
    plan, plan_json = written()
    assert abs(plan.duration_s - 30) <= 0.5
    assert verdict(plan, plan_json) is None
    assert mentioned(plan, BRIEF.must_mention) == ["lighthouse", "keeper", "lamp"]


def test_a_plan_that_cant_reach_its_target_passes_only_when_its_repairs_say_why():
    tight = PlanOptions(max_shot_s=3, max_shots=2)
    plan, _ = written(options=tight)
    assert plan.duration_s < 29 and any("target" in note for note in plan.repairs)
    assert verdict(plan, options=tight) is None
    silent = plan.model_copy(update={"repairs": []})
    assert "repairs don't say why" in verdict(silent, options=tight)


@pytest.mark.parametrize(
    "tamper, words",
    [
        (lambda plan, data: (plan, data + b" ", None), "content digest"),
        (lambda plan, data: (plan, data, receipt_for(plan, data, overrides={"plan": plan_info(shots=len(plan.shots) + 1)})), "misreports"),
        (lambda plan, data: (plan, data, receipt_for(plan, data, overrides={"plan": None, "video": VideoInfo(duration_s=30, width=1, height=1, fps=24, frames=1, audio=True)})),
         "certifies a video"),
    ],
)
def test_a_plan_that_doesnt_match_its_receipt_fails(tamper, words):
    plan, data = written()
    plan, data, receipt = tamper(plan, data)
    receipt = receipt or receipt_for(plan, encode_plan(plan))
    assert words in verdict(plan, data, receipt)


def test_a_plan_that_breaks_the_protocol_is_off_brief_or_unsafe_fails():
    # Written for another frame than the job's.
    other, _ = written(PLAN.model_copy(update={"resolution": "1080p"}))
    assert "protocol's rules" in verdict(other)
    # Canned: names none of the brief's terms.
    canned, _ = written(prompt="Wide shot; a person walks down a street. The camera pushes in.")
    assert "0 of the brief's 3 must-mention terms" in verdict(canned)
    # One of three terms is less than half.
    one = PlanBrief(BRIEF.brief, ("lighthouse", "tractor", "volcano"), 30.0)
    assert "1 of the brief's 3" in verdict(written()[0], brief=one)
    assert verdict(written()[0], brief=PlanBrief(BRIEF.brief, ("lighthouse", "tractor"), 30.0)) is None
    unsafe, _ = written(prompt=f"Wide shot; {BRIEF.brief} n.u.d.e keeper.")
    assert "content policy" in verdict(unsafe)


def test_only_a_planner_the_profile_runs_is_allowed():
    for planner, words in (
        ("ltx-2.5-distilled/bf16/1:text_encoder", "is not prompt_enhancer"),
        ("no-such-recipe/1:prompt_enhancer", "known precision recipe"),
        ("ltx-2.5-dev/bf16/1:prompt_enhancer", "not a recipe ltx-2.5-fast runs"),
        (MOCK_PLANNER, "mock planner outside a development network"),
    ):
        plan, _ = written(planner=planner)
        assert words in verdict(plan), planner
    mock, _ = written(planner=MOCK_PLANNER)
    assert verdict(mock, manifest=GoldenManifest(mock_quote_keys=["dev-key"])) is None
    old_prompt = written()[0]
    old_prompt = old_prompt.model_copy(update={"planner": old_prompt.planner.model_copy(update={"prompt_version": "plan/0"})})
    assert "prompt plan/0" in verdict(old_prompt)


def test_brief_sets_load_from_a_file_and_every_fallback_brief_is_a_valid_plan_target(tmp_path):
    path = tmp_path / "briefs.json"
    path.write_text(json.dumps([{"brief": "A heron fishing at dawn.", "must_mention": ["heron", "dawn"], "target_s": 12}]))
    [loaded] = load_briefs(path)
    assert loaded == PlanBrief("A heron fishing at dawn.", ("heron", "dawn"), 12.0) and pick_brief([loaded]) == loaded
    path.write_text(json.dumps([{"brief": "A heron.", "must_mention": [], "target_s": 12}]))
    with pytest.raises(ValueError):
        load_briefs(path)
    limits = FAST.limits
    assert all(limits.plan.min_target_s <= b.target_s <= limits.storyboard.max_total_s and len(b.must_mention) >= 2 for b in FALLBACK_BRIEFS)
