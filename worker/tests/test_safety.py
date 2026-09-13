"""The safety gate: obfuscation-resistant blocklist, pluggable classifiers that fail
closed, and the unchanged contract worker.py relies on. Classifiers here are fakes;
the real models are never downloaded in tests."""

from __future__ import annotations

import inspect
import logging
import subprocess
from pathlib import Path

import pytest

from kuno_worker import safety
from kuno_worker.backends.media_tools import ffmpeg_exe
from kuno_worker.safety import (
    Blocklist,
    SafetyGate,
    SafetyUnavailable,
    SafetyViolation,
    check_request,
    parse_qwen3guard,
)
from kuno_validator.canaries import FALLBACK_PROMPTS

SECRET = "ZEBRA-4412"


class FakeClassifier:
    name = "fake"

    def __init__(self, scores=None, error: Exception | None = None):
        self.scores, self.error, self.calls = scores or {}, error, 0

    def classify(self, text):
        self.calls += 1
        if self.error:
            raise self.error
        return self.scores


class FakeFrames:
    name = "fake-frames"

    def __init__(self, scores):
        self.scores, self.seen = scores, []

    def classify_frames(self, frames):
        self.seen = list(frames)
        return self.scores


@pytest.fixture(autouse=True)
def fresh_gate():
    safety.configure(SafetyGate())
    yield
    safety.configure(None)


def blocked(prompt: str, negative: str | None = None) -> bool:
    try:
        Blocklist().check(prompt, negative)
    except SafetyViolation:
        return True
    return False


# ---------------------------------------------------------------- contract


def test_the_worker_contract_is_unchanged():
    assert list(inspect.signature(check_request).parameters) == ["prompt", "negative_prompt"]
    assert inspect.signature(check_request).parameters["negative_prompt"].default is None
    assert issubclass(SafetyViolation, Exception) and not issubclass(SafetyUnavailable, SafetyViolation)
    assert check_request("A lighthouse in a storm") is None


def test_violation_messages_never_vary_with_the_request():
    with pytest.raises(SafetyViolation) as exc:
        check_request(f"csam {SECRET}")
    assert SECRET not in str(exc.value) and str(exc.value) == "request violates the acceptable use policy"


# ---------------------------------------------------------------- stage 1


@pytest.mark.parametrize(
    "prompt",
    [
        "csam",
        "CSAM compilation",
        "c.s.a.m video",
        "c s a m",
        "cs​am",  # zero-width space
        "сsаm",  # Cyrillic с and а
        "ｃｓａｍ",  # fullwidth
        "ch1ld p0rn",
        "chiiild pooorn",
        "kiddie-porn",
        "child_porn",
        "childpornography",
        "chíld pórn",
        "ch ild porn",
        "a 12 year old girl, nude",
        "12yo naked on a bed",
        "13-year-old, lingerie",
        "teen nsfw photoshoot",
        "loli hentai",
        "underage and explicit",
        "a schoolgirl undressing",
        "deepfake nude of a celebrity",
    ],
)
def test_obfuscated_abuse_is_blocked(prompt):
    assert blocked(prompt)


@pytest.mark.parametrize(
    "prompt",
    FALLBACK_PROMPTS
    + [
        "Kids building a sandcastle at the beach",
        "A 40 year old chef plating dessert",
        "A physics amateur explains lasers",
        "Sexy sports car commercial on a mountain road",
        "A nude marble statue in a museum, slow dolly",
        "A comic strip comes alive for kids",
        "A toddler laughing at bubbles, 5 years old",
        "Chicken breasts sizzling, kids waiting for dinner",
    ],
)
def test_ordinary_prompts_pass(prompt):
    assert not blocked(prompt)


def test_negative_prompts_may_list_what_to_avoid():
    assert not blocked("Children playing football in the park", "nsfw, nude, blurry")


def test_absolute_terms_in_negative_prompts_still_block():
    assert blocked("a playground", "csam")


def test_steering_a_minor_toward_nudity_through_the_negative_prompt_is_blocked():
    assert blocked("an 11 year old girl on a bed", "clothes, clothing")
    assert not blocked("an adult model on a runway", "clothes")


# ---------------------------------------------------------------- stage 2


def test_the_classifier_blocks_above_threshold():
    gate = SafetyGate(classifier=FakeClassifier({"sexual": 0.8}))
    with pytest.raises(SafetyViolation) as exc:
        gate.check_request("anything")
    assert exc.value.category == "sexual"


def test_scores_below_threshold_and_unlisted_categories_pass():
    gate = SafetyGate(classifier=FakeClassifier({"sexual": 0.2, "political": 1.0, "copyright": 1.0}))
    assert gate.check_request("anything") is None


def test_the_blocklist_runs_before_the_classifier():
    classifier = FakeClassifier()
    with pytest.raises(SafetyViolation):
        SafetyGate(classifier=classifier).check_request("csam")
    assert classifier.calls == 0


def test_the_classifier_only_sees_the_prompt():
    seen = []

    class Recording(FakeClassifier):
        def classify(self, text):
            seen.append(text)
            return {}

    SafetyGate(classifier=Recording()).check_request("a sunrise", "nsfw, gore")
    assert seen == ["a sunrise"]


def test_a_crashing_classifier_fails_closed_without_leaking(caplog):
    gate = SafetyGate(classifier=FakeClassifier(error=RuntimeError(f"tokenizer choked on {SECRET}")))
    with caplog.at_level(logging.DEBUG), pytest.raises(SafetyUnavailable) as exc:
        gate.check_request(f"prompt with {SECRET}")
    assert SECRET not in str(exc.value) and exc.value.__cause__ is None and exc.value.__suppress_context__
    assert SECRET not in caplog.text and "RuntimeError" in caplog.text


def test_a_classifier_that_failed_to_load_refuses_everything():
    with pytest.raises(SafetyUnavailable):
        SafetyGate(unavailable=True).check_request("a sunrise")


def test_require_classifier_refuses_work_without_one():
    with pytest.raises(SafetyUnavailable):
        SafetyGate(require_classifier=True).check_request("a sunrise")


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("Safety: Safe\nCategories: None", {}),
        ("Safety: Unsafe\nCategories: Violent", {"violence": 1.0}),
        ("Safety: Controversial\nCategories: Sexual Content or Sexual Acts", {"sexual": 0.5}),
        ("Safety: Unsafe\nCategories: Jailbreak, Suicide & Self-Harm", {"jailbreak": 1.0, "self_harm": 1.0}),
        ("Safety: Unsafe\nCategories: Something New", {"unknown_unsafe": 1.0}),
    ],
)
def test_qwen3guard_answers_are_parsed(output, expected):
    assert parse_qwen3guard(output) == expected


def test_default_thresholds_block_controversial_sexual_content_but_not_controversial_violence():
    with pytest.raises(SafetyViolation):
        SafetyGate(classifier=FakeClassifier(parse_qwen3guard("Safety: Controversial\nCategories: Sexual Content or Sexual Acts"))).check_request("x")
    SafetyGate(classifier=FakeClassifier(parse_qwen3guard("Safety: Controversial\nCategories: Violent"))).check_request("x")
    with pytest.raises(SafetyViolation):
        SafetyGate(classifier=FakeClassifier(parse_qwen3guard("Safety: Unsafe\nCategories: Something New"))).check_request("x")


def test_an_unparseable_guard_answer_fails_closed():
    class Garbled(FakeClassifier):
        def classify(self, text):
            return parse_qwen3guard("I cannot help with that")

    with pytest.raises(SafetyUnavailable):
        SafetyGate(classifier=Garbled()).check_request("x")


# ---------------------------------------------------------------- configuration


def test_no_configuration_means_blocklist_only_and_says_so(caplog):
    with caplog.at_level(logging.ERROR):
        gate = SafetyGate.from_env({})
    assert gate.classifier is None and not gate.unavailable and "KUNO_SAFETY_CLASSIFIER is not set" in caplog.text


def test_explicit_none_is_quiet(caplog):
    with caplog.at_level(logging.ERROR):
        gate = SafetyGate.from_env({"KUNO_SAFETY_CLASSIFIER": "none"})
    assert gate.classifier is None and caplog.text == ""


@pytest.mark.parametrize(
    "env",
    [
        {"KUNO_SAFETY_CLASSIFIER": "qwen3guard"},
        {"KUNO_SAFETY_CLASSIFIER": "qwen3guard", "KUNO_SAFETY_MODEL_PATH": "/nonexistent/weights"},
        {"KUNO_SAFETY_CLASSIFIER": "mystery", "KUNO_SAFETY_MODEL_PATH": "."},
    ],
)
def test_a_configured_classifier_that_cannot_load_fails_closed(env):
    gate = SafetyGate.from_env(env)
    assert gate.unavailable and gate.status()["classifier_unavailable"]
    with pytest.raises(SafetyUnavailable):
        gate.check_request("a sunrise")


def test_require_and_thresholds_come_from_the_environment():
    gate = SafetyGate.from_env({"KUNO_SAFETY_CLASSIFIER": "none", "KUNO_SAFETY_REQUIRE_CLASSIFIER": "1", "KUNO_SAFETY_THRESHOLDS": '{"violence": 0.4}'})
    assert gate.require_classifier and gate.thresholds["violence"] == 0.4 and gate.thresholds["sexual"] == 0.5


# ---------------------------------------------------------------- output frames


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> bytes:
    out = Path(tmp_path_factory.mktemp("frames")) / "clip.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=128x72:rate=12",
         "-t", "2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True, timeout=60,
    )
    return out.read_bytes()


def test_sampled_frames_reach_the_frame_classifier(clip):
    frames = FakeFrames({})
    SafetyGate(frame_classifier=frames, frames_to_sample=4).check_output(clip)
    assert 1 <= len(frames.seen) <= 4 and frames.seen[0].shape == (224, 224, 3)


def test_unsafe_frames_are_blocked(clip):
    with pytest.raises(SafetyViolation):
        SafetyGate(frame_classifier=FakeFrames({"sexual": 0.99})).check_output(clip)


def test_frames_that_cannot_be_decoded_fail_closed():
    with pytest.raises(SafetyUnavailable):
        SafetyGate(frame_classifier=FakeFrames({})).check_output(b"not a video")


def test_without_a_frame_classifier_output_checks_are_a_no_op():
    assert SafetyGate().check_output(b"anything") is None


# ---------------------------------------------------------------- through the real worker


def test_the_worker_reports_classifier_blocks_as_safety_blocked(worker_factory, caplog):
    worker, job = worker_factory(f"a quiet scene {SECRET}")
    safety.configure(SafetyGate(classifier=FakeClassifier({"sexual": 1.0})))
    with caplog.at_level(logging.DEBUG):
        worker.handle_job(job)
    code, message = worker.client.failures[0]
    assert code == "safety_blocked" and SECRET not in message and SECRET not in caplog.text


def test_an_unavailable_classifier_is_the_miners_failure_not_the_customers(worker_factory, caplog):
    worker, job = worker_factory(f"a quiet scene {SECRET}")
    safety.configure(SafetyGate(classifier=FakeClassifier(error=RuntimeError(SECRET))))
    with caplog.at_level(logging.DEBUG):
        worker.handle_job(job)
    code, message = worker.client.failures[0]
    assert code == "internal_error" and SECRET not in message and SECRET not in caplog.text


@pytest.fixture
def worker_factory(tmp_path):
    from test_privacy import ExplodingBackend, StubClient, sealed_job

    from kuno_protocol.attestation import MockTEE
    from kuno_protocol.crypto import generate_signing_key
    from kuno_protocol.devkit import DEV_IMAGE_DIGEST
    from kuno_worker.config import WorkerConfig
    from kuno_worker.worker import Worker

    def make(prompt: str):
        config = WorkerConfig(gateway_url="http://127.0.0.1:9", profiles=["ltx-2.5-fast"], image_digest=DEV_IMAGE_DIGEST, workdir=tmp_path / "work")
        worker = Worker(config, MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"*": ExplodingBackend()})
        worker.client = StubClient()
        worker.evidence = worker.attest(b"\x00" * 32)
        return worker, sealed_job(worker, prompt)

    return make
