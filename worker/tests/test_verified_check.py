"""kuno-verified-check on CPU: two runs of the same cases commit the same leaves, a divergence is named by the step
it happened at, runs that are not comparable are refused rather than judged, and the run file feeds the validator's
golden set. The GPU path is the same code with --backend real."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kuno_protocol.profiles import load_profiles
from kuno_validator.golden import default_cases as validator_cases
from kuno_validator.golden import golden_from_run
from kuno_worker import determinism

PROFILES = load_profiles()
DEV = "dev-cpu"


def run(tmp_path: Path, name: str, *extra: str) -> dict:
    out = tmp_path / f"{name}.json"
    assert determinism.main(["run", "--backend", "mock", "--out", str(out), *extra]) == 0
    return json.loads(out.read_text())


def first_run(tmp_path: Path, name: str = "a", profile: str = "ltx-2.5-fast", cases: int = 2) -> dict:
    return run(tmp_path, name, "--profile", profile, "--hardware-class", DEV, "--case-count", str(cases))


def test_a_run_writes_the_committed_leaves_of_every_case(tmp_path):
    doc = first_run(tmp_path)
    assert doc["schema"] == determinism.SCHEMA and doc["profile_id"] == "ltx-2.5-fast" and doc["hardware_class"] == DEV
    assert doc["comparison"] == "bitwise" and doc["backend"] == "mock"
    assert [case["name"] for case in doc["cases"]] == ["ltx-2.5-fast-0", "ltx-2.5-fast-1"]
    for record in doc["runs"]:
        assert record["outcome"] == "ok"
        # Every step of both stages, each with its sigma, plus the latent each stage started from: in order, no gaps.
        stages = PROFILES["ltx-2.5-fast"].verified.stage_steps
        assert len(record["leaves"]) == sum(stages) + len(stages)
        assert [leaf["index"] for leaf in record["leaves"]] == list(range(len(record["leaves"])))
        assert [leaf["index"] for leaf in record["leaves"] if leaf["kind"] == "init"] == [0, stages[0] + 1]
        assert [leaf["stage"] for leaf in record["leaves"]] == [0] * (stages[0] + 1) + [1] * (stages[1] + 1)
        assert len({leaf["latent"] for leaf in record["leaves"]}) == len(record["leaves"])
        assert record["root"] and record["model_digest"] and record["conditioning_digest"]


def test_the_same_cases_in_two_processes_commit_the_same_leaves(tmp_path):
    a = first_run(tmp_path)
    b = run(tmp_path, "b", "--cases", str(tmp_path / "a.json"))
    assert [case["name"] for case in b["cases"]] == [case["name"] for case in a["cases"]]
    assert b["hardware_class"] == DEV  # taken from the first run, so the second process cannot drift
    identical, differences, notes = determinism.compare_runs(a, b)
    assert identical and not differences
    # Leaves match under different roots: every job salts its own tree, which is what makes leaves comparable.
    assert {r["root"] for r in a["runs"]}.isdisjoint({r["root"] for r in b["runs"]})
    assert notes == ["2 of 2 cases: identical leaves under different roots, as expected (each job salts its own tree)"]
    assert determinism.main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0


def test_a_divergence_names_the_step_and_tells_the_noise_apart_from_the_denoiser(tmp_path):
    a = first_run(tmp_path)
    b = json.loads(json.dumps(a))
    b["runs"][0]["leaves"][0]["latent"] = "a" * 64
    b["runs"][1]["leaves"][4]["latent"] = "f" * 64
    identical, differences, _ = determinism.compare_runs(a, b)
    assert not identical
    assert differences[0] == f"ltx-2.5-fast-0: diverges at leaf 0 of {len(a['runs'][0]['leaves'])} (the seed's initial noise)"
    assert differences[1].startswith("ltx-2.5-fast-1: diverges at leaf 4 of") and "stage 0, sigma" in differences[1]
    (tmp_path / "b.json").write_text(json.dumps(b))
    assert determinism.main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 1


def test_a_differing_conditioning_blames_the_text_encoder_not_the_denoiser(tmp_path):
    a = first_run(tmp_path)
    b = json.loads(json.dumps(a))
    b["runs"][0]["conditioning_digest"] = "b" * 64
    b["runs"][0]["leaves"][3]["latent"] = "c" * 64
    _, differences, _ = determinism.compare_runs(a, b)
    assert differences == ["ltx-2.5-fast-0: the conditioning differs (the text encoder is not deterministic here)"]


def test_runs_that_cannot_be_compared_are_refused_rather_than_called_a_divergence(tmp_path):
    a = first_run(tmp_path)
    other_class = json.loads(json.dumps(a)) | {"hardware_class": "C1.rtx-pro-6000-bw-se.x1"}
    with pytest.raises(determinism.VerifiedCheckError, match="not comparable"):
        determinism.compare_runs(a, other_class)
    fewer = json.loads(json.dumps(a))
    fewer["cases"] = fewer["cases"][:1]
    with pytest.raises(determinism.VerifiedCheckError, match="different cases"):
        determinism.compare_runs(a, fewer)
    other_prompt = json.loads(json.dumps(a))
    other_prompt["cases"][0]["prompt"] = "something else entirely"
    with pytest.raises(determinism.VerifiedCheckError, match="differs between the runs"):
        determinism.compare_runs(a, other_prompt)
    other_weights = json.loads(json.dumps(a))
    other_weights["runs"][0]["model_digest"] = "d" * 64
    with pytest.raises(determinism.VerifiedCheckError, match="different weights"):
        determinism.compare_runs(a, other_weights)


def test_the_second_process_builds_its_backend_with_the_class_it_read_back(tmp_path, monkeypatch):
    """--cases carries the hardware class; without it reaching the backend, the second run would render unverified."""
    first_run(tmp_path, cases=1)
    seen = {}

    def real_backends(args, profile_ids, workdir):
        seen["hardware_class"] = args.hardware_class
        raise SystemExit("stop here: the backend would load weights")

    monkeypatch.setattr("kuno_worker.bench.real_backends", real_backends)
    with pytest.raises(SystemExit):
        determinism.main(["run", "--backend", "real", "--cases", str(tmp_path / "a.json"), "--out", str(tmp_path / "c.json")])
    assert seen["hardware_class"] == DEV


def test_a_class_without_verified_mode_is_refused_before_any_rendering(tmp_path, capsys):
    assert determinism.main(["run", "--backend", "mock", "--profile", "ltx-2.5-fast", "--hardware-class", "no-such-class",
                             "--out", str(tmp_path / "x.json")]) == 2
    assert "no verified hardware class" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def test_the_cases_are_the_ones_the_validator_computes_its_golden_set_from():
    for profile in PROFILES.values():
        if profile.verified is None:
            continue
        ours = [case.as_json() for case in determinism.default_cases(profile)]
        theirs = [case.model_dump(mode="json") for case in validator_cases(profile)]
        assert ours == theirs, profile.id


def test_a_good_run_becomes_a_golden_set_and_a_failed_one_does_not(tmp_path):
    a = first_run(tmp_path)
    golden = golden_from_run(a, image_digest="sha256:abc")
    assert golden.profile_id == "ltx-2.5-fast" and golden.hardware_class == DEV and golden.image_digest == "sha256:abc"
    assert golden.model_digest == a["runs"][0]["model_digest"] and golden.runtime == a["runtime"]
    assert [entry.leaf_digests for entry in golden.entries] == [[leaf["latent"] for leaf in r["leaves"]] for r in a["runs"]]
    broken = json.loads(json.dumps(a))
    broken["runs"][1] |= {"outcome": "error", "detail": "CUDA out of memory"}
    with pytest.raises(ValueError, match="did not finish every case"):
        golden_from_run(broken)


def test_repeats_in_one_process_are_compared_too(tmp_path, capsys):
    run(tmp_path, "r", "--profile", "ltx-2.5-fast", "--hardware-class", DEV, "--case-count", "1", "--repeats", "2")
    assert "repeats within this process: identical" in capsys.readouterr().out


def test_reference_profiles_run_their_own_mode(tmp_path):
    doc = first_run(tmp_path, profile="h3-reference", cases=1)
    assert doc["cases"][0]["params"]["mode"] == "reference_to_video"
    assert doc["runs"][0]["outcome"] == "ok" and doc["runs"][0]["leaves"]
