"""kuno-safety-check, the self-test that proves an image's safety classifiers load and answer. Fakes stand in
for the models here; the worker images run it against the real weights on CPU (image/CVM.md)."""

from __future__ import annotations

import json

import pytest

from kuno_worker.safety import SafetyGate
from kuno_worker.safety_check import PROMPT, main, run, synthetic_frame


class Prompt:
    name = "fake-guard"

    def __init__(self, scores=None, error: Exception | None = None):
        self.scores, self.error, self.seen = scores or {}, error, []

    def classify(self, text):
        self.seen.append(text)
        if self.error:
            raise self.error
        return self.scores


class Frames:
    input_size = 224

    def __init__(self, row=None, error: Exception | None = None, name: str = "fake-frames"):
        self.row, self.error, self.name, self.seen = row or {"sexual": 0.01, "suggestive": 0.02}, error, name, []

    def score_frames(self, frames):
        self.seen = list(frames)
        if self.error:
            raise self.error
        return [dict(self.row) for _ in frames]


def gate(**kwargs) -> SafetyGate:
    kwargs.setdefault("classifier", Prompt())
    kwargs.setdefault("frame_classifiers", [Frames(), Frames({"minor": 0.05}, name="fake-minor")])
    return SafetyGate(**kwargs)


def test_working_classifiers_pass_and_every_model_is_reported():
    prompt, frames, minor = Prompt({"violence": 0.0}), Frames(), Frames({"minor": 0.05}, name="fake-minor")
    report, problems = run(gate(classifier=prompt, frame_classifiers=[frames, minor]))
    assert problems == []
    assert prompt.seen == [PROMPT]
    assert len(frames.seen) == 1 and frames.seen[0].shape == (256, 256, 3) and frames.seen[0].dtype.name == "uint8"
    assert report["prompt"]["classifier"] == "fake-guard" and report["prompt"]["scores"] == {"violence": 0.0}
    assert [entry["classifier"] for entry in report["frame"]] == ["fake-frames", "fake-minor"]
    assert report["frame"][1]["scores"] == [{"minor": 0.05}]


def test_the_synthetic_frame_is_a_plain_gradient():
    frame = synthetic_frame(8)
    assert frame.shape == (8, 8, 3) and frame[0, 0].tolist() == [0, 0, 128] and frame[7, 7].tolist() == [255, 255, 128]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"classifier": None}, "no prompt classifier loaded"),
        ({"unavailable": True}, "no prompt classifier loaded"),
        ({"frame_classifiers": []}, "no frame classifier loaded"),
        ({"frame_unavailable": True}, "no frame classifier loaded"),
    ],
)
def test_a_missing_or_unloadable_classifier_fails_as_a_production_worker_would(kwargs, expected):
    report, problems = run(gate(**kwargs))
    assert any(expected in problem for problem in problems)
    assert "prompt" not in report


def test_a_classifier_that_blocks_the_benign_inputs_fails():
    _, problems = run(gate(classifier=Prompt({"sexual": 1.0})))
    assert problems == ["the benign prompt was blocked as sexual"]
    _, problems = run(gate(frame_classifiers=[Frames({"sexual": 0.9, "suggestive": 0.9}), Frames({"minor": 0.0}, name="fake-minor")]))
    assert problems == ["the synthetic frame was blocked as sexual"]


def test_a_classifier_that_crashes_or_returns_nonsense_fails():
    _, problems = run(gate(classifier=Prompt(error=RuntimeError("no weights"))))
    assert problems == ["prompt classifier fake-guard failed (RuntimeError: no weights)"]
    _, problems = run(gate(frame_classifiers=[Frames(error=OSError("bad file")), Frames({"minor": 0.0}, name="fake-minor")]))
    assert problems == ["frame classifier fake-frames failed (OSError: bad file)"]
    _, problems = run(gate(frame_classifiers=[Frames({"sexual": 7.0})]))
    assert len(problems) == 1 and "invalid score" in problems[0]


def test_the_command_prints_json_and_exits_1_without_classifiers(monkeypatch, capsys):
    monkeypatch.setenv("KUNO_SAFETY_CLASSIFIER", "none")
    for key in ("KUNO_SAFETY_FRAME_MODEL_PATH", "KUNO_SAFETY_MINOR_MODEL_PATH", "KUNO_SAFETY_THRESHOLDS", "KUNO_SAFETY_FRAME_THRESHOLDS"):
        monkeypatch.delenv(key, raising=False)
    assert main([]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False and len(report["problems"]) == 2


def test_malformed_settings_are_reported_as_a_failure(monkeypatch, capsys):
    monkeypatch.setenv("KUNO_SAFETY_CLASSIFIER", "none")
    monkeypatch.setenv("KUNO_SAFETY_THRESHOLDS", '{"sexual": 0.9}')
    assert main([]) == 1
    assert json.loads(capsys.readouterr().out)["problems"][0].startswith("invalid safety settings")


def test_the_command_exits_0_when_every_model_answers(monkeypatch, capsys):
    monkeypatch.setattr(SafetyGate, "from_env", classmethod(lambda cls, env=None: gate()))
    assert main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True and report["problems"] == [] and "load_seconds" in report
