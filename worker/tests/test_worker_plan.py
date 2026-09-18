"""Plan jobs in the worker: the brief checked before the planner sees it, the reply repaired, retried once and checked
shot by shot with one regeneration after a block, failures reported with the right code, the plan sealed padded to the
output key under its own label, a receipt with `plan` and no `video`, and the `plan/1` feature at registration.
Scripted planners replace the GPU; their replies include the raw outputs of the 2026-09-16 GPU spike."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from kuno_protocol.canonical import b64e, sha256_hex
from kuno_protocol.crypto import SenderSession, public_key_bytes
from kuno_protocol.plans import PLAN_OPTION, Plan, PlanOptions, PlanRevision, open_plan, plan_context, repair
from kuno_protocol.profiles import InputRole, Mode, load_profiles, storyboard_duration_s
from kuno_protocol.receipts import verify_receipt
from kuno_protocol.schemas import GenerationParams, MinerJob, SealedPayload, job_aad
from kuno_protocol.sealed_payload import seal_payload
from kuno_worker import safety, worker as worker_module
from kuno_worker.backends.base import Backend, PlanText
from kuno_worker.backends.ltx_resident import LtxResidentBackend
from kuno_worker.backends.mock import MOCK_PLANNER, MockBackend
from kuno_worker.safety import SafetyGate
from kuno_worker.worker import PLAN_FAILED_MESSAGE, PROMPT_BLOCKED, JobRejected

from test_worker_provenance import RecordingClient, make_worker

FAST = load_profiles()["ltx-2.5-fast"]
SPIKE = {
    (r["planner"], r["case"]): r
    for r in json.loads((Path(__file__).parents[2] / "protocol" / "tests" / "data" / "plan_spike_2026-09-16.json").read_text())["replies"]
}
ROASTERY, WATER, LIGHTHOUSE, GARBAGE = SPIKE["e2b", "roastery"], SPIKE["e2b", "water"], SPIKE["e2b", "lighthouse"], SPIKE["te12b", "roastery"]
PLANNER = "ltx-2.5-distilled/bf16/1:prompt_enhancer"
BLOCKED = "a naked woman on a bed"  # the shared content policy's "sexual" category


@pytest.fixture(autouse=True)
def fresh_gate():
    safety.configure(SafetyGate())
    yield
    safety.configure(None)


class PlanClient(RecordingClient):
    def __init__(self):
        super().__init__()
        self.failures: list[tuple[str, str]] = []
        self.strikes: list[bool] = []
        self.stages: list[str] = []

    def fail(self, _job_id, code, message, strike=True):
        self.failures.append((code, message))
        self.strikes.append(strike)

    def progress(self, _job_id, _value, stage):
        self.stages.append(stage)
        return False


class ScriptedPlanner(MockBackend):
    """The mock renderer with a planner that answers from a script, recording every chat and seed it was given."""

    def __init__(self, *replies: str, tokens: int = 400):
        super().__init__()
        self.replies, self.tokens = list(replies), tokens
        self.calls: list[tuple[list[dict], int, int]] = []

    def write_plan(self, task, messages, *, seed, max_new_tokens):
        self.calls.append((messages, seed, max_new_tokens))
        return PlanText(text=self.replies.pop(0), output_tokens=self.tokens, planner=PLANNER)


class RecordingClassifier:
    name = "recording"

    def __init__(self, blocks=lambda text: False):
        self.blocks, self.seen = blocks, []

    def classify(self, text):
        self.seen.append(text)
        return {"sexual": 1.0} if self.blocks(text) else {}


def plan_params(target: float = 30, aspect_ratio: str = "16:9", **update) -> GenerationParams:
    return GenerationParams(profile_id=FAST.id, mode=Mode.PLAN, duration_s=target, resolution="720p", aspect_ratio=aspect_ratio, fps=24, **update)


def plan_job(worker, brief: str = ROASTERY["brief"], *, params: GenerationParams | None = None, options: dict | None = None, seed: int = 7,
             **payload) -> tuple[MinerJob, bytes]:
    """A sealed plan job, padded as every sender seals, and the output key its client would open the plan with."""
    params = params or plan_params()
    job_id = str(uuid.uuid4())
    session = SenderSession(worker.identity.hpke_public)
    sealed = SealedPayload(prompt=brief, seed=seed, options={PLAN_OPTION: options} if options is not None else {}, **payload)
    ciphertext = seal_payload(session, sealed, job_aad(job_id, worker.identity.enclave_id, params, []))
    return MinerJob(job_id=job_id, params=params, enc=b64e(session.enc), ciphertext=b64e(ciphertext), input_blob_ids=[]), session.output_key


def planner_worker(tmp_path, backend: Backend):
    worker = make_worker(tmp_path, backend)
    worker.client = PlanClient()
    return worker


def failure(worker, job) -> tuple[str, str]:
    assert worker.handle_job(job) is None
    [failed] = worker.client.failures
    return failed


# ------------------------------------------------------------------ delivery


def test_the_mock_planner_delivers_a_sealed_padded_plan_with_a_plan_receipt(tmp_path):
    worker = planner_worker(tmp_path, MockBackend())
    brief = "A 20-second explainer for the app Sip, ending on the slogan 'Sip. Stay fresh.'"
    job, output_key = plan_job(worker, brief, params=plan_params(20, "9:16"))
    receipt = worker.process(job)

    [sealed] = worker.client.uploads
    plan, data = open_plan(output_key, job.job_id, sealed)
    assert plan.profile_id == FAST.id and plan.aspect_ratio == "9:16" and abs(plan.duration_s - 20) <= 0.5
    assert plan.planner.model == MOCK_PLANNER and any('"Sip. Stay fresh."' in shot.prompt for shot in plan.shots)
    assert plan.duration_s == storyboard_duration_s(FAST, plan.shot_specs(), 24)

    body = receipt.body
    assert body.video is None and body.step_commitment is None
    assert (body.plan.shots, body.plan.duration_s, body.plan.planner, body.plan.prompt_version) == (len(plan.shots), plan.duration_s, MOCK_PLANNER, "plan/1")
    assert body.content_digest == sha256_hex(data) and (body.output_digest, body.output_bytes) == (sha256_hex(sealed), len(sealed))
    assert "video" not in body.model_dump(mode="json") and body.plan.output_tokens > 0
    assert verify_receipt(receipt, public_key_bytes(worker.identity.signing_key))
    # Padded like a request (4 KiB bucket), then PADMÉ: every plan of up to about 4 KB is the same size.
    assert len(sealed) == 4386
    assert worker.client.stages == ["decrypted", "planning", "checking", "sealing"]


def test_the_early_closed_gpu_reply_is_repaired_and_delivered_as_written(tmp_path):
    backend = ScriptedPlanner(ROASTERY["raw"])
    worker = planner_worker(tmp_path, backend)
    job, output_key = plan_job(worker, ROASTERY["brief"], options={"max_shot_s": 11})
    receipt = worker.process(job)
    plan, _ = open_plan(output_key, job.job_id, worker.client.uploads[0])
    context = plan_context(FAST, job.params, PlanOptions(max_shot_s=11))
    assert plan == repair(ROASTERY["raw"], context, planner=PLANNER, brief=ROASTERY["brief"]).plan
    assert receipt.body.plan.output_tokens == 400
    [(messages, seed, max_new_tokens)] = backend.calls
    assert seed == 7 and max_new_tokens == 2048 and [m["role"] for m in messages] == ["system", "user"]
    assert "whole number from 2 to 11" in messages[0]["content"] and messages[1]["content"] == f"Brief: {ROASTERY['brief']}"


def test_without_max_shot_s_shots_are_planned_to_this_workers_envelope(tmp_path, monkeypatch):
    """A 4090 serves 720p 16:9 at 24 fps up to 7 s, so that is the longest shot it plans when the client names none."""
    from kuno_protocol import torch_verified

    monkeypatch.setattr(torch_verified, "apply_determinism", lambda settings: {"torch": "stub"})  # a verified class pins first
    written: list = []

    class Loaded:
        def write_text(self, messages, **kwargs):
            written.append((messages, kwargs))
            return ROASTERY["raw"], 451

    backend = LtxResidentBackend(None, tmp_path / "work", loader=lambda _p: Loaded(), hardware_class="O1.rtx-4090-24gb.x1.int8", host_ram_gib=128)
    worker = planner_worker(tmp_path, backend)
    job, output_key = plan_job(worker)
    receipt = worker.process(job)
    [(messages, kwargs)] = written
    assert "whole number from 2 to 7." in messages[0]["content"]
    assert kwargs == {"seed": 7, "max_new_tokens": 2048, "do_sample": True, "temperature": 0.7, "top_p": 0.95, "top_k": 64}
    plan, _ = open_plan(output_key, job.job_id, worker.client.uploads[0])
    assert max(shot.duration_s for shot in plan.shots) <= 7
    # No load plan on the stand-in, so the planner is named from the class's recipe.
    assert plan.planner.model == receipt.body.plan.planner == "ltx-2.5-distilled/int8-wo/1:prompt_enhancer"


# ------------------------------------------------------------------ retry and failure codes


def test_an_unparseable_reply_is_retried_once_with_the_problem_as_a_user_turn(tmp_path):
    backend = ScriptedPlanner(GARBAGE["raw"], ROASTERY["raw"])
    worker = planner_worker(tmp_path, backend)
    receipt = worker.process(plan_job(worker)[0])
    (first, first_seed, _), (retry, retry_seed, _) = backend.calls
    assert retry[:2] == first and retry[2] == {"role": "assistant", "content": GARBAGE["raw"]}
    assert retry[3]["role"] == "user" and retry[3]["content"].startswith("Your reply had no JSON object.")
    assert (first_seed, retry_seed) == (7, 8) and receipt.body.plan.output_tokens == 800


def test_a_reply_that_is_unparseable_twice_fails_as_plan_failed_and_uploads_nothing(tmp_path):
    worker = planner_worker(tmp_path, ScriptedPlanner(GARBAGE["raw"], '{"title": "T", "shots": [{"prompt": "only one"}]}'))
    assert failure(worker, plan_job(worker)[0]) == ("plan_failed", PLAN_FAILED_MESSAGE)
    assert worker.client.uploads == [] and worker.client.completed == []


def test_a_missing_brief_quote_is_retried_and_named_in_repairs_when_the_retry_drops_it_too(tmp_path):
    backend = ScriptedPlanner(WATER["raw"], WATER["raw"])
    worker = planner_worker(tmp_path, backend)
    job, output_key = plan_job(worker, WATER["brief"], params=plan_params(20, "9:16"), options={"max_shot_s": 11})
    worker.process(job)
    [_, (retry, _, _)] = backend.calls
    assert '"Sip. Stay fresh."' in retry[-1]["content"]
    plan, _ = open_plan(output_key, job.job_id, worker.client.uploads[0])
    assert plan.repairs[-1] == 'the brief\'s quoted words "Sip. Stay fresh." are in no shot'


def test_a_retry_that_fixes_the_quote_is_the_plan_delivered(tmp_path):
    fixed = WATER["raw"].replace("Stay refreshed and energized throughout your day.", "Sip. Stay fresh.")
    worker = planner_worker(tmp_path, ScriptedPlanner(WATER["raw"], fixed))
    job, output_key = plan_job(worker, WATER["brief"], params=plan_params(20, "9:16"))
    worker.process(job)
    plan, _ = open_plan(output_key, job.job_id, worker.client.uploads[0])
    assert '"Sip. Stay fresh."' in plan.shots[-1].prompt and not any("quoted" in r for r in plan.repairs)


def test_a_refusal_is_safety_blocked_without_a_retry(tmp_path):
    backend = ScriptedPlanner('{"refusal": "cannot plan this brief"}', ROASTERY["raw"])
    worker = planner_worker(tmp_path, backend)
    assert failure(worker, plan_job(worker)[0]) == ("safety_blocked", PROMPT_BLOCKED)
    assert len(backend.calls) == 1


# ------------------------------------------------------------------ safety


def test_the_brief_style_and_instruction_are_checked_before_the_planner_runs(tmp_path, monkeypatch):
    for payload in ({"brief": BLOCKED}, {"options": {"style": BLOCKED}}):
        backend = ScriptedPlanner(ROASTERY["raw"])
        worker = planner_worker(tmp_path, backend)
        job, _ = plan_job(worker, **payload)
        assert failure(worker, job) == ("safety_blocked", PROMPT_BLOCKED) and backend.calls == []
    seen = []
    monkeypatch.setattr(worker_module, "check_request", lambda text, negative=None: seen.append(text))
    worker = planner_worker(tmp_path, ScriptedPlanner(ROASTERY["raw"]))
    worker.process(plan_job(worker, options={"style": "35mm film, warm", "max_shot_s": 11})[0])
    plan = repair(ROASTERY["raw"], plan_context(FAST, plan_params(), PlanOptions(max_shot_s=11)), planner=PLANNER, brief=ROASTERY["brief"]).plan
    # The brief, the style, then every shot's model prompt (scene + prompt), as the video model will see it.
    assert seen == [ROASTERY["brief"], "35mm film, warm", *plan.model_prompts()]


def test_a_blocked_shot_prompt_is_written_again_once_from_the_next_seeds(tmp_path):
    classifier = RecordingClassifier(blocks=lambda text: "coffee beans gently shifting" in text or "pile of dark, roasted coffee beans" in text)
    safety.configure(SafetyGate(classifier=classifier))
    backend = ScriptedPlanner(ROASTERY["raw"], LIGHTHOUSE["raw"].replace('"duration_s": 6', '"duration_s": 4'))
    worker = planner_worker(tmp_path, backend)
    job, output_key = plan_job(worker, options={"max_shot_s": 11})
    receipt = worker.process(job)
    assert [seed for _, seed, _ in backend.calls] == [7, 9]
    plan, _ = open_plan(output_key, job.job_id, worker.client.uploads[0])
    assert "lighthouse" in plan.scene and receipt.body.plan.output_tokens == 800
    assert worker.client.stages.count("checking") == 1  # a regeneration shows no stage of its own


def test_a_second_block_fails_the_job_as_safety_blocked(tmp_path):
    safety.configure(SafetyGate(classifier=RecordingClassifier(blocks=lambda text: "\n\n" in text)))  # every shot prompt
    backend = ScriptedPlanner(ROASTERY["raw"], ROASTERY["raw"])
    worker = planner_worker(tmp_path, backend)
    assert failure(worker, plan_job(worker)[0]) == ("safety_blocked", PROMPT_BLOCKED)
    assert len(backend.calls) == 2 and worker.client.uploads == []


def test_titles_notes_and_beats_are_held_to_the_content_policy(tmp_path):
    blocked = json.loads(ROASTERY["raw"].replace("]} ,", "],"))
    blocked["notes"] = BLOCKED
    backend = ScriptedPlanner(json.dumps(blocked), ROASTERY["raw"])
    worker = planner_worker(tmp_path, backend)
    job, output_key = plan_job(worker)
    worker.process(job)
    plan, _ = open_plan(output_key, job.job_id, worker.client.uploads[0])
    assert len(backend.calls) == 2 and BLOCKED not in plan.notes

    beat = json.loads(ROASTERY["raw"].replace("]} ,", "],"))
    beat["shots"][2]["beat"] = BLOCKED
    worker = planner_worker(tmp_path, ScriptedPlanner(json.dumps(beat), json.dumps(beat)))
    assert failure(worker, plan_job(worker)[0]) == ("safety_blocked", PROMPT_BLOCKED)


# ------------------------------------------------------------------ strikes: who wrote what was blocked


def test_a_blocked_brief_style_or_instruction_is_the_customers_and_a_strike(tmp_path):
    first = planner_worker(tmp_path, ScriptedPlanner(ROASTERY["raw"]))
    earlier = delivered(first, *plan_job(first, options={"max_shot_s": 11}))
    revise = PlanRevision(plan=earlier, instruction=BLOCKED, shots=[3]).model_dump(mode="json")
    for payload in ({"brief": BLOCKED}, {"options": {"style": BLOCKED}}, {"brief": "", "options": {"max_shot_s": 11, "revise": revise}}):
        backend = ScriptedPlanner(ROASTERY["raw"])
        worker = planner_worker(tmp_path, backend)
        job, _ = plan_job(worker, **payload)
        assert failure(worker, job) == ("safety_blocked", PROMPT_BLOCKED) and worker.client.strikes == [True]
        assert backend.calls == []


def test_what_the_planner_wrote_blocked_twice_is_no_strike(tmp_path):
    # A shot prompt, through the prompt check.
    safety.configure(SafetyGate(classifier=RecordingClassifier(blocks=lambda text: "\n\n" in text)))  # every shot prompt
    worker = planner_worker(tmp_path, ScriptedPlanner(ROASTERY["raw"], ROASTERY["raw"]))
    assert failure(worker, plan_job(worker)[0]) == ("safety_blocked", PROMPT_BLOCKED) and worker.client.strikes == [False]

    # A beat, through the content policy.
    safety.configure(SafetyGate())
    beat = json.loads(ROASTERY["raw"].replace("]} ,", "],"))
    beat["shots"][2]["beat"] = BLOCKED
    worker = planner_worker(tmp_path, ScriptedPlanner(json.dumps(beat), json.dumps(beat)))
    assert failure(worker, plan_job(worker)[0]) == ("safety_blocked", PROMPT_BLOCKED) and worker.client.strikes == [False]

    # A first block the second plan fixes costs nothing at all: the plan is delivered.
    worker = planner_worker(tmp_path, ScriptedPlanner(json.dumps(beat), ROASTERY["raw"]))
    worker.process(plan_job(worker)[0])
    assert worker.client.failures == [] and len(worker.client.uploads) == 1


def test_the_planners_refusal_of_a_brief_that_passed_the_checks_is_no_strike(tmp_path):
    backend = ScriptedPlanner('{"refusal": "cannot plan this brief"}')
    worker = planner_worker(tmp_path, backend)
    assert failure(worker, plan_job(worker)[0]) == ("safety_blocked", PROMPT_BLOCKED) and worker.client.strikes == [False]
    assert len(backend.calls) == 1


def test_blocked_text_in_a_revisions_earlier_plan_is_the_customers_and_strikes_before_the_planner_runs(tmp_path):
    """A revision gives parts of the earlier plan back byte-identical, so the worker checks it as the customer's before the
    planner runs: a block after that is always the planner's text."""
    first = planner_worker(tmp_path, ScriptedPlanner(ROASTERY["raw"]))
    earlier = delivered(first, *plan_job(first, options={"max_shot_s": 11}))
    shot = earlier.shots[0].model_copy(update={"prompt": f"{earlier.shots[0].prompt} {BLOCKED}."})
    for edited in (earlier.model_copy(update={"notes": BLOCKED}), earlier.model_copy(update={"shots": [shot, *earlier.shots[1:]]})):
        backend = ScriptedPlanner(ROASTERY["raw"])
        worker = planner_worker(tmp_path, backend)
        revise = PlanRevision(plan=edited, instruction="make it darker", shots=[3]).model_dump(mode="json")
        job, _ = plan_job(worker, "", options={"max_shot_s": 11, "revise": revise})
        assert failure(worker, job) == ("safety_blocked", PROMPT_BLOCKED) and worker.client.strikes == [True]
        assert backend.calls == []

    # The earlier plan passes; the planner's rewrite of shot 3 is blocked twice: the planner's text, no strike.
    rewritten = json.loads(ROASTERY["raw"].replace("]} ,", "],"))
    rewritten["shots"][2]["prompt"] = f"Close-up shot; {BLOCKED}."
    rewritten["shots"][2]["duration_s"] = 11
    backend = ScriptedPlanner(json.dumps(rewritten), json.dumps(rewritten))
    worker = planner_worker(tmp_path, backend)
    revise = PlanRevision(plan=earlier, instruction="make it darker", shots=[3]).model_dump(mode="json")
    job, _ = plan_job(worker, "", options={"max_shot_s": 11, "revise": revise})
    assert failure(worker, job) == ("safety_blocked", PROMPT_BLOCKED) and worker.client.strikes == [False]
    assert len(backend.calls) == 2


# ------------------------------------------------------------------ refusals before planning


@pytest.mark.parametrize(("payload", "code"), [
    ({"brief": "x" * 4001}, "prompt_too_long"),
    ({"options": {"style": "x" * 501}}, "prompt_too_long"),
    ({"negative_prompt": "blurry"}, "unsupported_option"),
    ({"options": {"temperature": 2}}, "bad_payload"),
    ({"options": {"max_shot_s": 1}}, "bad_payload"),
    ({"brief": "   "}, "bad_payload"),
])
def test_requests_a_plan_cannot_take_are_refused_before_the_planner_runs(tmp_path, payload, code):
    backend = ScriptedPlanner(ROASTERY["raw"])
    worker = planner_worker(tmp_path, backend)
    job, _ = plan_job(worker, **payload)
    assert failure(worker, job)[0] == code and backend.calls == []


def test_a_backend_without_a_planner_refuses_before_decrypting(tmp_path):
    class Renderer(Backend):
        def generate(self, task, progress):
            raise AssertionError("never")

    worker = planner_worker(tmp_path, Renderer())
    job = MinerJob(job_id=str(uuid.uuid4()), params=plan_params(), enc="AA", ciphertext="AA", input_blob_ids=[])
    assert failure(worker, job) == ("internal_error", "This worker's backend does not write plans.")
    with pytest.raises(JobRejected, match="invalid_params|does not accept"):
        worker.process(MinerJob(job_id=str(uuid.uuid4()), params=plan_params(input_roles=[InputRole.FIRST_FRAME]), enc="AA", ciphertext="AA", input_blob_ids=[]))


# ------------------------------------------------------------------ revisions


def delivered(worker, job, output_key) -> Plan:
    worker.process(job)
    return open_plan(output_key, job.job_id, worker.client.uploads[-1])[0]


def test_a_revision_rewrites_only_the_listed_shots(tmp_path):
    first = planner_worker(tmp_path, ScriptedPlanner(ROASTERY["raw"]))
    earlier = delivered(first, *plan_job(first, options={"max_shot_s": 11}))

    rewritten = json.loads(ROASTERY["raw"].replace("]} ,", "],"))
    rewritten["title"] = "Something else"
    rewritten["shots"][2]["prompt"] = "Close-up shot; a darker, slower pour of beans into the cooler. The sound of beans rattles."
    rewritten["shots"][2]["duration_s"] = 11
    backend = ScriptedPlanner(json.dumps(rewritten))
    worker = planner_worker(tmp_path, backend)
    revise = PlanRevision(plan=earlier, instruction="make it darker", shots=[3]).model_dump(mode="json")
    plan = delivered(worker, *plan_job(worker, "", options={"max_shot_s": 11, "revise": revise}))
    assert plan.title == earlier.title and plan.scene == earlier.scene
    assert [plan.shots[i] for i in (0, 1, 3, 4)] == [earlier.shots[i] for i in (0, 1, 3, 4)]
    assert plan.shots[2].prompt.startswith("Close-up shot; a darker") and abs(plan.duration_s - 30) <= 0.5
    [(messages, _, _)] = backend.calls
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"] and messages[1]["content"] == "Brief: "


def test_a_revision_of_a_plan_for_another_frame_is_refused(tmp_path):
    first = planner_worker(tmp_path, ScriptedPlanner(ROASTERY["raw"]))
    earlier = delivered(first, *plan_job(first, options={"max_shot_s": 11}))
    worker = planner_worker(tmp_path, ScriptedPlanner(ROASTERY["raw"]))
    revise = PlanRevision(plan=earlier, shots=[1]).model_dump(mode="json")
    job, _ = plan_job(worker, params=plan_params(30, "9:16"), options={"revise": revise})
    code, message = failure(worker, job)
    assert code == "bad_payload" and "different profile, size" in message


# ------------------------------------------------------------------ registration


def test_registration_advertises_plan_1_only_when_the_backend_writes_plans(tmp_path):
    class Client:
        def __init__(self):
            self.extra = []

        def nonce(self):
            return b"\x00" * 32

        def register(self, evidence, miner_hotkey, capacity, hotkey_proof=None, **extra):
            self.extra.append(extra)

    class Renderer(Backend):
        def generate(self, task, progress):
            raise AssertionError("never")

    for backend, features in ((MockBackend(), ["plan/1"]), (Renderer(), [])):
        worker = make_worker(tmp_path, backend)
        worker.client = Client()
        worker._refresh_certificate = lambda: None
        worker.register()
        assert worker.features == features and worker.client.extra[0].get("features") == (features or None)


def test_the_gateway_client_sends_features_only_when_there_are_some():
    import httpx

    from kuno_protocol.attestation import MockTEE, build_evidence
    from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key
    from kuno_worker.gateway_client import GatewayClient

    bodies = []
    client = GatewayClient("http://gw.test", generate_signing_key(), "e" * 32,
                           transport=httpx.MockTransport(lambda r: bodies.append(json.loads(r.content)) or httpx.Response(200, json={})))
    key, (_, hpke) = generate_signing_key(), generate_hpke_keypair()
    evidence = build_evidence(MockTEE(key, "sha256:img"), b"\x00" * 32, hpke, public_key_bytes(key), "sha256:img", [FAST.id])
    client.register(evidence, "5Miner", 1)
    client.register(evidence, "5Miner", 1, features=["plan/1"])
    assert "features" not in bodies[0] and bodies[1]["features"] == ["plan/1"]


# ------------------------------------------------------------------ tools


def test_kuno_plan_shows_the_planners_chat_and_kuno_bench_never_measures_a_plan(tmp_path, monkeypatch, capsys):
    from kuno_worker import plan as plan_tool
    from kuno_worker.bench import bench_mode

    task = plan_tool.build_task(FAST, plan_tool.example_task(FAST, Mode.PLAN, duration_s=30), tmp_path, prompt=ROASTERY["brief"], seed=3)
    request = plan_tool.plan_request(task)
    assert request["messages"][1] == {"role": "user", "content": f"Brief: {ROASTERY['brief']}"}
    assert request["generate"] == {"max_new_tokens": 2048, "seed": 3, "do_sample": True, "temperature": 0.7, "top_p": 0.95, "top_k": 64}
    assert (request["suggested_shots"], request["suggested_shot_s"], request["max_shot_s"]) == (5, 7, 20)
    # A plan's duration is a target, whose minimum is its own.
    assert plan_tool.example_task(FAST, Mode.PLAN).duration_s == FAST.limits.plan.min_target_s

    monkeypatch.setattr("sys.argv", ["kuno-plan", FAST.id, "plan", "--duration", "45", "--json"])
    plan_tool.main()
    shown = json.loads(capsys.readouterr().out)
    assert (shown["mode"], shown["price_usd"], shown["rendered_frames"]) == ("plan", 0.1, 0)

    assert bench_mode(FAST) is Mode.TEXT_TO_VIDEO
    assert bench_mode(FAST.model_copy(update={"modes": [Mode.PLAN, Mode.STORYBOARD, Mode.KEYFRAMES]})) is Mode.KEYFRAMES
