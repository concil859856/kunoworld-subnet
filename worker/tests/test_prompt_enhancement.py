"""Prompt enhancement is a step of its own, checked before anything renders: the backend's enhancer rewrites the prompt,
the worker runs the safety gate on the rewrite exactly as on the customer's prompt, and the render conditions on the
checked text with the pipeline's own enhancement off. Storyboards refuse the option, a backend without an enhancer
renders the prompt as sent, and verified mode commits to and retains the prompt actually rendered. Stand-ins replace
the enhancer, the GPU and the classifiers."""

from __future__ import annotations

import logging
import uuid

import numpy as np
import pytest

from kuno_protocol import toy_denoiser
from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import SenderSession
from kuno_protocol.profiles import Mode, load_profiles, shot_prompt, storyboard_duration_s
from kuno_protocol.schemas import GenerationParams, MinerJob, SealedPayload, ShotPrompt, ShotSpec, job_aad
from kuno_protocol.verified import latent_digest
from kuno_worker import safety, worker as worker_module
from kuno_worker.backends.base import ENHANCE_PROMPT_OPTION
from kuno_worker.backends.ltx import LtxPaths, build_command
from kuno_worker.backends.ltx_resident import LtxResidentBackend, build_call
from kuno_worker.backends.media_tools import BackendError, ffmpeg_exe
from kuno_worker.backends.mock import MockBackend
from kuno_worker.plan import build_task, example_task, storyboard_plan
from kuno_worker.safety import SafetyGate
from kuno_worker.verified import RetentionStore
from kuno_worker.worker import PROMPT_BLOCKED, JobRejected

from test_worker_provenance import RecordingClient, make_worker

pytestmark = pytest.mark.skipif(ffmpeg_exe() is None, reason="ffmpeg encodes the rendered video")

FAST = load_profiles()["ltx-2.5-fast"]
SECRET = "ZEBRA-4412"
PROMPT = "A lighthouse keeper lights the lamp at dusk"
ENHANCED = (
    "A wide shot frames a stone lighthouse on a cliff at dusk as its keeper climbs the spiral stair and strikes a match; "
    "the lamp flares and its beam sweeps the darkening sea while waves break below and gulls call."
)
BLOCKED = "a naked woman on a bed"  # the shared content policy's "sexual" category
SCENE, SHOTS = "An old harbor at golden hour", ["The boat pulls away from the dock", "A gull lands on a post"]


@pytest.fixture(autouse=True)
def fresh_gate():
    safety.configure(SafetyGate())
    yield
    safety.configure(None)


class FailureClient(RecordingClient):
    def __init__(self):
        super().__init__()
        self.failures: list[tuple[str, str]] = []

    def fail(self, _job_id, code, message):
        self.failures.append((code, message))


class EnhancingBackend(MockBackend):
    """The mock renderer with a stand-in enhancer. `events` records each step in order, with the prompt it was given."""

    prompt_enhancement = True

    def __init__(self, enhanced: str = ENHANCED, events: list | None = None, **kwargs):
        super().__init__(**kwargs)
        self.enhanced = enhanced
        self.events = events if events is not None else []
        self.rendered = []

    def enhance_prompt(self, task):
        self.events.append(("enhance", task.prompt))
        return self.enhanced

    def generate(self, task, progress):
        self.events.append(("render", task.prompt))
        self.rendered.append(task)
        return super().generate(task, progress)


class SpyBackend(MockBackend):
    def __init__(self):
        super().__init__()
        self.rendered = []

    def generate(self, task, progress):
        self.rendered.append(task)
        return super().generate(task, progress)


class RecordingClassifier:
    name = "recording"

    def __init__(self, blocks: str | None = None, fails_on: str | None = None):
        self.blocks, self.fails_on, self.seen = blocks, fails_on, []

    def classify(self, text):
        self.seen.append(text)
        if text == self.fails_on:
            raise RuntimeError(f"classifier crashed on {text!r}")
        return {"sexual": 1.0} if text == self.blocks else {}


def text_params() -> GenerationParams:
    return GenerationParams(profile_id=FAST.id, mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24)


def storyboard_params() -> GenerationParams:
    shots = [ShotSpec(duration_s=2, join="fresh"), ShotSpec(duration_s=2, join="continue")]
    return GenerationParams(
        profile_id=FAST.id, mode=Mode.STORYBOARD, duration_s=storyboard_duration_s(FAST, shots, 24), resolution="720p",
        aspect_ratio="16:9", fps=24, shots=shots,
    )


def sealed(worker, payload: SealedPayload, params: GenerationParams | None = None) -> MinerJob:
    params = params or text_params()
    job_id = str(uuid.uuid4())
    session = SenderSession(worker.identity.hpke_public)
    ciphertext = session.seal(payload.model_dump_json().encode(), job_aad(job_id, worker.identity.enclave_id, params, []))
    return MinerJob(job_id=job_id, params=params, enc=b64e(session.enc), ciphertext=b64e(ciphertext), input_blob_ids=[])


def enhanced_job(worker, prompt: str = PROMPT) -> MinerJob:
    return sealed(worker, SealedPayload(prompt=prompt, seed=7, options={ENHANCE_PROMPT_OPTION: True}))


def rejected(worker, job) -> JobRejected:
    with pytest.raises(JobRejected) as exc:
        worker.process(job)
    return exc.value


# ---------------------------------------------------------------- the checked step


def test_the_enhanced_prompt_is_checked_then_rendered_in_place_of_the_customers(tmp_path, monkeypatch):
    events: list = []
    backend = EnhancingBackend(events=events)
    worker = make_worker(tmp_path, backend)
    monkeypatch.setattr(worker_module, "check_request", lambda prompt, negative=None: events.append(("check", prompt)))
    worker.process(enhanced_job(worker))
    # The customer's prompt is checked before the enhancer sees it, and the enhancer's before anything renders.
    assert events == [("check", PROMPT), ("enhance", PROMPT), ("check", ENHANCED), ("render", ENHANCED)]
    [task] = backend.rendered
    assert task.prompt == ENHANCED and ENHANCE_PROMPT_OPTION not in task.options and task.seed == 7


def test_the_gateway_cannot_tell_from_progress_that_enhancement_was_asked_for(tmp_path):
    class ProgressClient(RecordingClient):
        def __init__(self):
            super().__init__()
            self.stages: list[str] = []

        def progress(self, _job_id, _value, stage):
            self.stages.append(stage)
            return False

    seen = {}
    for options in ({}, {ENHANCE_PROMPT_OPTION: True}):
        backend = EnhancingBackend()
        worker = make_worker(tmp_path, backend)
        worker.client = ProgressClient()
        worker.process(sealed(worker, SealedPayload(prompt=PROMPT, seed=7, options=options)))
        # The worker's own stages; the backend's are throttled by the clock, and its rendering is the same either way.
        seen[bool(options)] = [stage for stage in worker.client.stages if stage in ("decrypted", "generating", "checking", "sealing")]
    assert seen[True] == seen[False] == ["decrypted", "generating", "checking", "sealing"]
    assert backend.events == [("enhance", PROMPT), ("render", ENHANCED)]


def test_the_resident_backend_renders_the_enhanced_prompt_with_the_pipelines_enhancement_off(tmp_path):
    class LoadedLtx:
        """runtimes.LtxAdapter's surface: the enhancer and the render record what they were called with."""

        def __init__(self):
            self.enhancements, self.renders = [], []

        def enhance_prompt(self, call):
            self.enhancements.append(dict(call))
            return ENHANCED

        def __call__(self, **call):
            self.renders.append(call)
            return {"videos": [[np.full((48, 64, 3), i * 5, dtype=np.uint8) for i in range(48)]], "audio": None, "sampling_rate": 48000}

    loaded = LoadedLtx()
    backend = LtxResidentBackend(None, tmp_path / "work", loader=lambda _profile: loaded)
    worker = make_worker(tmp_path, backend)
    worker.process(enhanced_job(worker))
    [enhancement], [render] = loaded.enhancements, loaded.renders
    assert (enhancement["prompt"], enhancement["seed"], enhancement["pipeline"]) == (PROMPT, 7, "text")
    assert render["prompt"] == ENHANCED and render["enable_prompt_enhancement"] is False
    assert backend.store.loads == 1


def test_no_render_call_enhances_whatever_the_options_say(tmp_path):
    options = {ENHANCE_PROMPT_OPTION: True}
    for mode in (Mode.TEXT_TO_VIDEO, Mode.IMAGE_TO_VIDEO):
        task = build_task(FAST, example_task(FAST, mode), tmp_path, options=options)
        for item in task.inputs:
            item.save(tmp_path)
        assert build_call(task)["enable_prompt_enhancement"] is False
        # The cold backend's CLI would enhance inside its own process, where nothing checks the result.
        argv, _, _ = build_command(task, LtxPaths(tmp_path), tmp_path, tmp_path / "out.mp4")
        assert "--enhance-prompt" not in argv
    board = build_task(FAST, example_task(FAST, Mode.STORYBOARD), tmp_path, options=options)
    assert all(shot["call"]["enable_prompt_enhancement"] is False for shot in storyboard_plan(board)["shots"])


# ---------------------------------------------------------------- blocks


def test_an_enhanced_prompt_the_content_policy_blocks_is_safety_blocked_before_anything_renders(tmp_path, caplog):
    backend = EnhancingBackend(enhanced=f"{BLOCKED}, {SECRET}")
    worker = make_worker(tmp_path, backend)
    worker.client = FailureClient()
    with caplog.at_level(logging.DEBUG):
        worker.handle_job(enhanced_job(worker))
    assert backend.events == [("enhance", PROMPT)] and backend.rendered == []
    assert worker.client.failures == [("safety_blocked", PROMPT_BLOCKED)] and worker.client.uploads == []
    assert SECRET not in caplog.text

    # Word for word what a blocked customer prompt gets: the gateway can't tell that enhancement was asked for.
    plain = EnhancingBackend()
    worker = make_worker(tmp_path, plain)
    error = rejected(worker, enhanced_job(worker, prompt=BLOCKED))
    assert (error.code, error.message) == ("safety_blocked", PROMPT_BLOCKED)
    assert plain.events == []  # a blocked prompt is never enhanced


def test_the_prompt_classifier_judges_the_enhanced_prompt(tmp_path):
    classifier = RecordingClassifier()
    safety.configure(SafetyGate(classifier=classifier))
    worker = make_worker(tmp_path, EnhancingBackend())
    worker.process(enhanced_job(worker))
    assert classifier.seen == [PROMPT, ENHANCED]

    safety.configure(SafetyGate(classifier=RecordingClassifier(blocks=ENHANCED)))
    backend = EnhancingBackend()
    worker = make_worker(tmp_path, backend)
    assert rejected(worker, enhanced_job(worker)).code == "safety_blocked"
    assert backend.rendered == []


def test_a_classifier_that_cannot_judge_the_enhanced_prompt_fails_the_job_as_the_miners(tmp_path, caplog):
    safety.configure(SafetyGate(classifier=RecordingClassifier(fails_on=ENHANCED)))
    backend = EnhancingBackend()
    worker = make_worker(tmp_path, backend)
    worker.client = FailureClient()
    with caplog.at_level(logging.DEBUG):
        worker.handle_job(enhanced_job(worker))
    [(code, message)] = worker.client.failures
    assert code == "internal_error" and backend.rendered == []
    assert "lighthouse" not in message and "lighthouse" not in caplog.text


def test_an_enhancer_that_writes_nothing_is_the_workers_failure(tmp_path):
    backend = EnhancingBackend(enhanced="  \n")
    worker = make_worker(tmp_path, backend)
    worker.client = FailureClient()
    worker.handle_job(enhanced_job(worker))
    assert worker.client.failures == [("internal_error", "Generation failed inside the worker.")] and backend.rendered == []


def test_the_frame_check_errs_toward_blocking_on_what_the_enhanced_prompt_names(tmp_path, monkeypatch):
    worker = make_worker(tmp_path, EnhancingBackend(enhanced="A young girl and her grandfather fish off the pier at dawn"))
    seen = []
    monkeypatch.setattr(worker_module, "check_output", lambda video, signals, shot_frames=None: seen.append(signals))
    worker.process(enhanced_job(worker))
    assert [signals.mentions_minor for signals in seen] == [True]


# ---------------------------------------------------------------- where it doesn't apply


def test_storyboards_refuse_prompt_enhancement_before_anything_runs(tmp_path, monkeypatch):
    backend = EnhancingBackend()
    worker = make_worker(tmp_path, backend)
    checked = []
    monkeypatch.setattr(worker_module, "check_request", lambda prompt, negative=None: checked.append(prompt))
    payload = SealedPayload(prompt=SCENE, seed=7, shots=[ShotPrompt(prompt=p) for p in SHOTS], options={ENHANCE_PROMPT_OPTION: True})
    error = rejected(worker, sealed(worker, payload, storyboard_params()))
    assert error.code == "unsupported_option" and "storyboards" in error.message
    assert checked == [] and backend.events == []

    # Without the option the same storyboard renders every shot as written.
    worker.process(sealed(worker, payload.model_copy(update={"options": {}}), storyboard_params()))
    assert backend.rendered[0].shot_prompts == [shot_prompt(SCENE, p) for p in SHOTS]
    assert [kind for kind, _ in backend.events] == ["render"]

    resident = LtxResidentBackend(None, tmp_path / "work", loader=lambda _profile: pytest.fail("nothing may load"))
    with pytest.raises(BackendError):
        resident.enhance_prompt(build_task(FAST, example_task(FAST, Mode.STORYBOARD), tmp_path))


def test_a_backend_without_an_enhancer_renders_the_prompt_as_sent(tmp_path):
    backend = SpyBackend()
    worker = make_worker(tmp_path, backend)
    worker.process(enhanced_job(worker))
    [task] = backend.rendered
    assert task.prompt == PROMPT and task.options == {ENHANCE_PROMPT_OPTION: True}


def test_a_profile_without_an_enhancer_never_asks_for_one(tmp_path):
    backend = EnhancingBackend()
    worker = make_worker(tmp_path, backend)
    worker.profiles[FAST.id] = FAST.model_copy(update={"limits": FAST.limits.model_copy(update={"prompt_enhancer": False})})
    worker.process(enhanced_job(worker))
    assert backend.events == [("render", PROMPT)]


# ---------------------------------------------------------------- verified mode


def test_verified_mode_commits_to_and_retains_the_prompt_actually_rendered(tmp_path):
    store = RetentionStore()
    backend = EnhancingBackend(retention=store)
    worker = make_worker(tmp_path, backend)
    verified = FAST.verified.model_copy(update={"retention_checkpoint_every": 3})
    # Every third leaf is kept, so reading the others recomputes them from the retained context: the prompt.
    worker.profiles[FAST.id] = FAST.model_copy(update={"verified": verified})
    job = enhanced_job(worker)
    receipt = worker.process(job)

    record = store.record(job.job_id)
    assert receipt.body.step_commitment is not None and record is not None
    assert record.transcript.conditioning_digest == toy_denoiser.conditioning_digest(ENHANCED) != toy_denoiser.conditioning_digest(PROMPT)
    for index, leaf in enumerate(record.leaves):
        assert latent_digest(store.latents(job.job_id, index)) == leaf.latent


# ---------------------------------------------------------------- the adapter (diffusers)


class PlacedEnhancer:
    """The enhancer module: the adapter keeps it in host RAM between uses (test_enhancer_placement.py)."""

    def to(self, device):
        return self


class EnhancerPipeline:
    """Where LtxAdapter.enhance_prompt meets diffusers: `LTX2Pipeline.enhance_prompt` returns one text per prompt."""

    def __init__(self, enhancer: object | None = PlacedEnhancer()):
        self.prompt_enhancer, self.processor, self.calls = enhancer, "processor", []

    def enhance_prompt(self, **kwargs):
        self.calls.append(kwargs)
        return [ENHANCED]


def test_the_adapter_enhances_as_diffusers_would_inside_the_render_call(tmp_path):
    pytest.importorskip("diffusers")
    Image = pytest.importorskip("PIL.Image")
    from diffusers.pipelines.ltx2.utils import LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT, LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT

    from kuno_worker.backends.runtimes import LtxAdapter

    text, condition = EnhancerPipeline(), EnhancerPipeline()
    adapter = LtxAdapter({"text": text, "condition": condition}, device="cpu")
    t2v = build_call(build_task(FAST, example_task(FAST, Mode.TEXT_TO_VIDEO), tmp_path, prompt=PROMPT, seed=9))
    assert adapter.enhance_prompt(t2v) == ENHANCED
    assert text.calls == [{"prompt": PROMPT, "system_prompt": LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT, "seed": 9, "image": None}]

    # LTX2ConditionPipeline enhances with the first condition's image and the image-to-video instructions.
    task = build_task(FAST, example_task(FAST, Mode.IMAGE_TO_VIDEO), tmp_path, prompt=PROMPT, seed=9)
    Image.new("RGB", (16, 12), "navy").save(task.inputs[0].save(tmp_path), format="PNG")
    adapter.enhance_prompt(build_call(task))
    [call] = condition.calls
    assert call["system_prompt"] == LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT and call["image"].size == (16, 12)

    # Without the dedicated enhancer diffusers would fall back to the text encoder, which LTX-2.5 didn't train for it.
    with pytest.raises(RuntimeError, match="no prompt_enhancer"):
        LtxAdapter({"text": EnhancerPipeline(enhancer=None)}, device="cpu").enhance_prompt(t2v)
