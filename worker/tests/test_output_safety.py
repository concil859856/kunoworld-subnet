"""Frame-level output safety: where frames are sampled, how the clip policy decides, that
every failure fails closed without leaking, and that the worker never seals, signs or
uploads a blocked video. Classifiers are fakes; real weights are only ever exercised by
scripts/benchmark_frame_safety.py, on benign synthetic clips."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pytest

from kuno_protocol.mp4 import probe
from kuno_worker import safety
from kuno_worker.backends.media_tools import ffmpeg_exe
from kuno_worker.backends.mock import MockBackend
from kuno_worker.safety import SafetyConfigError, SafetyGate, SafetyUnavailable, SafetyViolation
from kuno_worker.safety_frames import (
    FramePolicy,
    RequestSignals,
    load_frame_models,
    map_labels,
    plan_frame_indices,
    sample_frames,
)

SECRET = "OKAPI-9051"
BLOCKED = ("safety_blocked", "The video was blocked by the content policy.")


class Scorer:
    """A per-frame fake. Returns `row` for every frame unless it is told to misbehave."""

    def __init__(self, row=None, *, error: Exception | None = None, input_size: int = 224, drop: bool = False, log=None):
        self.row, self.error, self.input_size, self.drop, self.log = row or {}, error, input_size, drop, log
        self.name, self.calls, self.seen = "fake-scorer", 0, []

    def score_frames(self, frames):
        self.calls += 1
        self.seen = list(frames)
        if self.log is not None:
            self.log.append("scored")
        if self.error:
            raise self.error
        rows = [dict(self.row) for _ in frames]
        return rows[:-1] if self.drop else rows


class PromptClassifier:
    name = "fake-prompt"

    def classify(self, text):
        return {}


@pytest.fixture(autouse=True)
def fresh_gate():
    safety.configure(SafetyGate())
    yield
    safety.configure(None)


def _ffmpeg(*args: str) -> None:
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True, capture_output=True, timeout=60)


@pytest.fixture(scope="module")
def indexed_clip(tmp_path_factory) -> tuple[bytes, list[float]]:
    """30 lossless frames where frame n has luma 16 + 7n, plus each frame's decoded mean brightness."""
    out = Path(tmp_path_factory.mktemp("indexed")) / "indexed.mp4"
    _ffmpeg("-f", "lavfi", "-i", "color=c=black:s=64x64:r=24:d=1.25,format=yuv420p,geq=lum='16+7*N':cb=128:cr=128",
            "-frames:v", "30", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(out))
    raw = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True, timeout=60,
    ).stdout
    means = [float(np.frombuffer(raw[i : i + 64 * 64], dtype=np.uint8).mean()) for i in range(0, len(raw), 64 * 64)]
    return out.read_bytes(), means


@pytest.fixture(scope="module")
def mock_video() -> bytes:
    """What the mock backend renders for an ordinary 2 s, 24 fps text-to-video job."""
    from kuno_protocol.profiles import Mode, load_profiles
    from kuno_protocol.schemas import GenerationParams
    from kuno_worker.backends.base import GenerationTask

    profile = load_profiles()["ltx-2.5-fast"]
    params = GenerationParams(profile_id=profile.id, mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24)
    width, height = profile.size_for(params.resolution, params.aspect_ratio)
    task = GenerationTask(job_id="j", profile=profile, params=params, prompt="a lighthouse", negative_prompt=None, seed=7, width=width, height=height)
    return MockBackend().generate(task, lambda *_: None).data


def _nearest(means: list[float], frame: np.ndarray) -> int:
    value = frame.mean()  # grey frames: RGB mean equals luma mean
    return min(range(len(means)), key=lambda i: abs(means[i] - value))


# ---------------------------------------------------------------- sampling


def test_the_plan_spreads_frames_evenly_and_includes_both_ends():
    assert plan_frame_indices(240, 10) == [0, 27, 53, 80, 106, 133, 159, 186, 212, 239]
    assert plan_frame_indices(5, 10) == [0, 1, 2, 3, 4]
    assert plan_frame_indices(1, 10) == [0]
    assert plan_frame_indices(0, 10) == []


def test_sampled_frames_come_from_the_planned_positions(indexed_clip):
    video, means = indexed_clip
    assert len(means) == 30
    frames = sample_frames(video, 6)
    assert [_nearest(means, f) for f in frames] == plan_frame_indices(30, 6) == [0, 6, 12, 17, 23, 29]


def test_positions_inside_the_decoded_tail_are_mapped_correctly(indexed_clip):
    video, means = indexed_clip
    frames = sample_frames(video, 30)
    assert [_nearest(means, f) for f in frames] == list(range(30))


def test_the_mock_backends_video_is_sampled_at_the_requested_count_and_size(mock_video):
    assert probe(mock_video).frames == 48
    frames = sample_frames(mock_video, 10, size=448)
    assert len(frames) == 10 and all(f.shape == (448, 448, 3) and f.dtype == np.uint8 for f in frames)
    assert len({f.tobytes() for f in frames}) > 1  # distinct moments, not one frame repeated


def test_the_gate_samples_at_the_largest_model_input_size(mock_video):
    small, large = Scorer({"sexual": 0.0, "minor": 0.0}), Scorer({"sexual": 0.0}, input_size=448)
    SafetyGate(frame_classifiers=[small, large]).check_output(mock_video)
    assert len(small.seen) == len(large.seen) == 10 and small.seen[0].shape == (448, 448, 3)


# ---------------------------------------------------------------- policy


@pytest.mark.parametrize(
    ("policy", "rows", "signals", "expected"),
    [
        (FramePolicy(), [{"sexual": 0.6, "minor": 0.0}], None, "sexual"),
        (FramePolicy(), [{"sexual": 0.01, "suggestive": 0.02, "minor": 0.95}], None, None),  # kids at the beach
        (FramePolicy(), [{"sexual": 0.3, "minor": 0.5}], None, "sexual_minors"),  # below the explicit bar, still blocked
        (FramePolicy(allow_explicit=True), [{"sexual": 0.95, "minor": 0.05}], None, None),
        (FramePolicy(allow_explicit=True), [{"sexual": 0.2, "minor": 0.4}], None, "sexual_minors"),
        (FramePolicy(allow_explicit=True), [{"sexual": 0.05, "suggestive": 0.55, "minor": 0.5}], None, "sexual_minors"),
        # A minor in one frame and sexual content in another block together.
        (FramePolicy(allow_explicit=True), [{"sexual": 0.0, "minor": 0.9}, {"sexual": 0.3, "minor": 0.0}], None, "sexual_minors"),
        # Without a minor-presence model a minor is assumed.
        (FramePolicy(allow_explicit=True), [{"sexual": 0.2}], None, "sexual_minors"),
        (FramePolicy(), [{"sexual": 0.05}], None, None),
        # A prompt naming a minor overrides a detector that saw none.
        (FramePolicy(), [{"sexual": 0.2, "minor": 0.0}], RequestSignals(mentions_minor=True), "sexual_minors"),
        (FramePolicy(), [{"sexual": 0.2, "minor": 0.0}], RequestSignals(mentions_minor=False), None),
    ],
)
def test_the_clip_policy(policy, rows, signals, expected):
    assert policy.decide(rows, signals) == expected


@pytest.mark.parametrize("bad", [float("nan"), 1.5, -0.1, "0.3", float("inf")])
def test_invalid_scores_are_rejected(bad):
    with pytest.raises(ValueError):
        FramePolicy().decide([{"sexual": bad}])


def test_policy_settings_come_from_the_environment():
    policy = FramePolicy.from_env({"KUNO_SAFETY_ALLOW_NSFW": "1", "KUNO_SAFETY_FRAME_THRESHOLDS": '{"minor": 0.2}'})
    assert policy.allow_explicit and policy.minor == 0.2 and policy.minor_sexual == FramePolicy.minor_sexual
    assert not FramePolicy.from_env({}).allow_explicit  # strict by default
    for bad in ('{"sexy": 0.3}', '{"minor": 2}', "[0.3]"):
        with pytest.raises(ValueError):
            FramePolicy.from_env({"KUNO_SAFETY_FRAME_THRESHOLDS": bad})


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["normal", "nsfw"], [(), ("sexual", "suggestive")]),  # Falconsai
        (["neutral", "low", "medium", "high"], [(), ("suggestive",), ("sexual", "suggestive"), ("sexual", "suggestive")]),  # Freepik
        (["NSFW", "SFW"], [("sexual", "suggestive"), ()]),  # Marqo
    ],
)
def test_known_classifier_labels_map_to_categories(labels, expected):
    assert map_labels(labels) == expected


def test_unknown_labels_fail_instead_of_being_ignored():
    with pytest.raises(ValueError):
        map_labels(["normal", "cat"])
    with pytest.raises(ValueError):
        map_labels(["normal", "sexy"])  # nothing maps to sexual
    assert map_labels(["normal", "cat"], {"cat": ["sexual"]}) == [(), ("sexual",)]


# ---------------------------------------------------------------- the gate


def test_explicit_output_is_blocked_with_the_fixed_message(indexed_clip):
    with pytest.raises(SafetyViolation) as exc:
        SafetyGate(frame_classifiers=[Scorer({"sexual": 0.9, "minor": 0.0})]).check_output(indexed_clip[0])
    assert exc.value.category == "sexual" and str(exc.value) == "request violates the acceptable use policy"


def test_clean_output_passes(indexed_clip):
    scorer = Scorer({"sexual": 0.001, "suggestive": 0.002, "minor": 0.01})
    assert SafetyGate(frame_classifiers=[scorer]).check_output(indexed_clip[0]) is None
    assert len(scorer.seen) == 10


def test_other_frame_categories_use_the_ordinary_thresholds(indexed_clip):
    with pytest.raises(SafetyViolation) as exc:
        SafetyGate(frame_classifiers=[Scorer({"sexual": 0.0, "minor": 0.0, "self_harm": 0.7})]).check_output(indexed_clip[0])
    assert exc.value.category == "self_harm"


def test_a_crashing_frame_model_fails_closed_without_leaking_frames(indexed_clip, caplog):
    class Leaky(Scorer):
        def score_frames(self, frames):
            raise RuntimeError(f"{SECRET} {frames[0].tobytes()[:48].hex()}")

    frame_hex = sample_frames(indexed_clip[0], 10)[0].tobytes()[:48].hex()
    with caplog.at_level(logging.DEBUG), pytest.raises(SafetyUnavailable) as exc:
        SafetyGate(frame_classifiers=[Leaky()]).check_output(indexed_clip[0])
    assert exc.value.__cause__ is None and exc.value.__suppress_context__
    assert SECRET not in caplog.text and frame_hex not in caplog.text and "RuntimeError" in caplog.text


@pytest.mark.parametrize("scorer", [Scorer({"sexual": 0.0}, drop=True), Scorer({"sexual": float("nan")})])
def test_a_scorer_that_skips_frames_or_returns_nonsense_fails_closed(indexed_clip, scorer):
    with pytest.raises(SafetyUnavailable):
        SafetyGate(frame_classifiers=[scorer]).check_output(indexed_clip[0])


def test_an_undecodable_video_fails_closed(caplog):
    with caplog.at_level(logging.DEBUG), pytest.raises(SafetyUnavailable):
        SafetyGate(frame_classifiers=[Scorer()]).check_output(SECRET.encode() * 50)
    assert SECRET not in caplog.text


def test_frame_models_that_failed_to_load_refuse_before_generation():
    gate = SafetyGate(frame_unavailable=True)
    with pytest.raises(SafetyUnavailable):
        gate.check_request("a lighthouse in a storm")
    with pytest.raises(SafetyUnavailable):
        gate.check_output(b"anything")


def test_require_classifier_needs_a_frame_model_too():
    gate = SafetyGate(require_classifier=True, classifier=PromptClassifier())
    with pytest.raises(SafetyUnavailable):
        gate.check_request("a lighthouse in a storm")
    with pytest.raises(SafetyUnavailable):
        gate.check_output(b"anything")
    assert len(gate.startup_errors()) == 1
    assert SafetyGate(require_classifier=True, classifier=PromptClassifier(), frame_classifiers=[Scorer()]).startup_errors() == []


def test_without_frame_models_the_output_check_is_a_no_op():
    assert SafetyGate().check_output(b"not even a video") is None


def test_request_signals_carry_booleans_not_text():
    gate = SafetyGate()
    assert gate.request_signals(f"Kids building a sandcastle {SECRET}").mentions_minor
    assert not gate.request_signals("A lighthouse in a storm", "children, people").mentions_minor
    assert SECRET not in repr(gate.request_signals(f"kids {SECRET}"))


# ---------------------------------------------------------------- configuration


def test_no_frame_model_is_logged_once_and_outputs_pass(caplog):
    with caplog.at_level(logging.ERROR):
        gate = SafetyGate.from_env({})
    assert gate.frame_classifiers == [] and not gate.frame_unavailable
    assert caplog.text.count("KUNO_SAFETY_FRAME_MODEL_PATH is not set") == 1
    assert load_frame_models({}) == ([], [])


def test_an_explicit_opt_out_stays_quiet_about_frames_too(caplog):
    with caplog.at_level(logging.ERROR):
        SafetyGate.from_env({"KUNO_SAFETY_CLASSIFIER": "none"})
    assert caplog.text == ""


@pytest.mark.parametrize(
    "env",
    [
        {"KUNO_SAFETY_FRAME_MODEL_PATH": "/nonexistent/nsfw"},
        {"KUNO_SAFETY_MINOR_MODEL_PATH": "/nonexistent/clip"},  # a minor detector alone cannot judge sexual content
        {"KUNO_SAFETY_FRAME_MODEL_PATH": "{labels}"},  # a checkpoint whose labels mean nothing to the policy
    ],
)
def test_configured_frame_models_that_cannot_load_fail_closed(env, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "vit", "id2label": {"0": "cat", "1": "dog"}}')
    env = {k: v.replace("{labels}", str(tmp_path)) for k, v in env.items()}
    gate = SafetyGate.from_env({"KUNO_SAFETY_CLASSIFIER": "none", **env})
    assert gate.frame_unavailable and gate.status()["frame_classifier_unavailable"]
    with pytest.raises(SafetyUnavailable):
        gate.check_request("a lighthouse in a storm")


def test_malformed_frame_settings_are_rejected():
    with pytest.raises(ValueError):
        SafetyGate.from_env({"KUNO_SAFETY_CLASSIFIER": "none", "KUNO_SAFETY_FRAMES": "1"})
    assert SafetyGate.from_env({"KUNO_SAFETY_CLASSIFIER": "none", "KUNO_SAFETY_FRAMES": "16"}).frames_to_sample == 16


def test_a_worker_that_requires_classifiers_refuses_to_start_without_them(monkeypatch):
    import os

    for key in [k for k in os.environ if k.startswith("KUNO_SAFETY_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("KUNO_SAFETY_REQUIRE_CLASSIFIER", "1")
    safety.configure(None)
    with pytest.raises(SafetyConfigError, match="KUNO_SAFETY_FRAME_MODEL_PATH") as exc:
        safety.default_gate()
    assert isinstance(exc.value, ValueError)  # kuno-worker turns ValueError into a clean exit


# ---------------------------------------------------------------- through the real worker


class _Spy(MockBackend):
    def __init__(self):
        super().__init__()
        self.generated = 0

    def generate(self, task, progress):
        self.generated += 1
        return super().generate(task, progress)


@pytest.fixture
def mock_worker(tmp_path):
    from test_privacy import StubClient

    from kuno_protocol.attestation import MockTEE
    from kuno_protocol.crypto import generate_signing_key
    from kuno_protocol.devkit import DEV_IMAGE_DIGEST
    from kuno_worker.config import WorkerConfig
    from kuno_worker.worker import Worker

    class RecordingClient(StubClient):
        def __init__(self):
            super().__init__()
            self.uploads = []

        def upload_blob(self, *args):
            self.uploads.append(args)
            return "0" * 32

    config = WorkerConfig(gateway_url="http://127.0.0.1:9", profiles=["ltx-2.5-fast"], image_digest=DEV_IMAGE_DIGEST, workdir=tmp_path / "work")
    backend = _Spy()
    worker = Worker(config, MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"*": backend})
    worker.client = RecordingClient()
    worker.evidence = worker.attest(b"\x00" * 32)
    worker.backend_spy = backend
    return worker


def _job(worker, prompt):
    from test_privacy import sealed_job

    return sealed_job(worker, prompt)


def _spy_on_provenance(worker, monkeypatch, events):
    original = worker._embed_provenance

    def embed(rendered, draft):
        events.append("embedded")
        return original(rendered, draft)

    monkeypatch.setattr(worker, "_embed_provenance", embed)


def test_a_blocked_video_is_never_sealed_signed_or_uploaded(mock_worker, monkeypatch, caplog):
    events: list[str] = []
    scorer = Scorer({"sexual": 0.97, "minor": 0.0}, log=events)
    safety.configure(SafetyGate(frame_classifiers=[scorer]))
    _spy_on_provenance(mock_worker, monkeypatch, events)
    with caplog.at_level(logging.DEBUG):
        receipt = mock_worker.handle_job(_job(mock_worker, f"a quiet harbour at dusk {SECRET}"))
    assert receipt is None and mock_worker.client.failures == [BLOCKED]
    assert mock_worker.client.uploads == [] and mock_worker.client.completed == []
    assert events == ["scored"]  # provenance never ran
    assert SECRET not in caplog.text


def test_an_allowed_video_is_checked_before_provenance_and_delivered(mock_worker, monkeypatch):
    events: list[str] = []
    scorer = Scorer({"sexual": 0.001, "minor": 0.0}, log=events)
    safety.configure(SafetyGate(frame_classifiers=[scorer]))
    _spy_on_provenance(mock_worker, monkeypatch, events)
    receipt = mock_worker.handle_job(_job(mock_worker, "a quiet harbour at dusk"))
    assert receipt is not None and mock_worker.client.failures == []
    assert events == ["scored", "embedded"] and len(mock_worker.client.uploads) == 1
    assert len(scorer.seen) == 10


def test_a_broken_frame_model_is_the_miners_failure(mock_worker, caplog):
    safety.configure(SafetyGate(frame_classifiers=[Scorer(error=RuntimeError(f"frame decode {SECRET}"))]))
    with caplog.at_level(logging.DEBUG):
        mock_worker.handle_job(_job(mock_worker, f"a quiet harbour {SECRET}"))
    (code, message), = mock_worker.client.failures
    assert code == "internal_error" and SECRET not in message and SECRET not in caplog.text
    assert mock_worker.client.uploads == []


def test_an_unloadable_frame_model_refuses_before_spending_gpu_time(mock_worker):
    safety.configure(SafetyGate(frame_unavailable=True))
    mock_worker.handle_job(_job(mock_worker, "a quiet harbour"))
    assert mock_worker.backend_spy.generated == 0
    assert [code for code, _ in mock_worker.client.failures] == ["internal_error"] and mock_worker.client.uploads == []


def test_a_prompt_naming_a_minor_lowers_the_bar_for_the_frames(mock_worker):
    # Borderline scores with no minor detected: allowed for an adult scene, blocked once the prompt names children.
    safety.configure(SafetyGate(frame_classifiers=[Scorer({"sexual": 0.2, "suggestive": 0.2, "minor": 0.0})]))
    assert mock_worker.handle_job(_job(mock_worker, "a couple dancing on a rooftop")) is not None
    assert mock_worker.handle_job(_job(mock_worker, "kids dancing on a rooftop")) is None
    assert mock_worker.client.failures == [BLOCKED] and len(mock_worker.client.uploads) == 1
