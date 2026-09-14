"""Worker start-up rules: open-tier workers need a hotkey; production (TDX) workers need their safety classifiers."""

from __future__ import annotations

import os

import pytest

from kuno_protocol.attestation import OpenTEE
from kuno_protocol.hotkey import Sr25519Signer
from kuno_worker import safety
from kuno_worker.backends.mock import MockBackend
from kuno_worker.config import WorkerConfig
from kuno_worker.main import build_tee, check_hotkey, check_safety
from kuno_worker.safety import SafetyConfigError, SafetyGate
from kuno_worker.worker import Worker


def config(**fields) -> WorkerConfig:
    return WorkerConfig(gateway_url="http://127.0.0.1:9", profiles=["ltx-2.5-fast"], **fields)


class Prompt:
    name = "prompt"

    def classify(self, text):
        return {}


class Frames:
    name = "frames"

    def score_frames(self, frames):
        return [{} for _ in frames]


def test_open_mode_builds_a_quote_less_provider_and_refuses_to_start_without_a_hotkey():
    tee = build_tee(config(tee="open"))
    assert isinstance(tee, OpenTEE) and tee.quote(b"\0" * 64) == b"" and tee.gpu_evidence(b"\0" * 32) is None
    with pytest.raises(SystemExit, match="KUNO_TEE=open needs the miner's hotkey"):
        check_hotkey(config(tee="open"), None)
    check_hotkey(config(tee="open"), Sr25519Signer.from_seed(os.urandom(32)))
    check_hotkey(config(tee="mock"), None)  # dev networks are unchanged
    with pytest.raises(SystemExit, match="'tdx', 'open' or 'mock'"):
        build_tee(config(tee="sgx"))


def test_an_open_tier_worker_object_needs_its_hotkey_and_no_gateway_certificate():
    with pytest.raises(ValueError, match="open-tier worker"):
        Worker(config(tee="open"), OpenTEE(), {"*": MockBackend()})
    signer = Sr25519Signer.from_seed(os.urandom(32))
    with pytest.raises(ValueError, match="KUNO_PROVENANCE=off"):
        Worker(config(tee="open", provenance="c2pa"), OpenTEE(), {"*": MockBackend()}, hotkey=signer)
    worker = Worker(config(tee="open"), OpenTEE(), {"*": MockBackend()}, hotkey=signer)
    evidence = worker.attest(os.urandom(32))
    assert (evidence.tee, evidence.quote, evidence.gpu_evidence) == ("open", "", None)
    assert worker.miner_hotkey == signer.ss58_address


def test_a_tdx_worker_refuses_to_start_without_prompt_and_frame_classifiers():
    with pytest.raises(SafetyConfigError, match="prompt classifier.*frame classifier"):
        check_safety(config(tee="tdx"), SafetyGate())
    with pytest.raises(SafetyConfigError, match="frame classifier"):
        check_safety(config(tee="tdx"), SafetyGate(classifier=Prompt()))
    with pytest.raises(SafetyConfigError, match="prompt classifier"):
        check_safety(config(tee="tdx"), SafetyGate(classifier=Prompt(), frame_classifiers=[Frames()], unavailable=True))
    assert isinstance(SafetyConfigError("x"), ValueError)  # kuno-worker turns it into a clean exit

    gate = SafetyGate(classifier=Prompt(), frame_classifiers=[Frames()])
    check_safety(config(tee="tdx"), gate)
    assert gate.require_classifier  # and it fails closed if a classifier later disappears


def test_mock_and_open_workers_keep_the_permissive_default(monkeypatch):
    for tee in ("mock", "open"):
        gate = SafetyGate()
        check_safety(config(tee=tee), gate)
        assert not gate.require_classifier
    # KUNO_SAFETY_REQUIRE_CLASSIFIER keeps working on its own.
    assert SafetyGate(require_classifier=True).startup_errors()
    assert SafetyGate().startup_errors() == []
    safety.configure(None)
